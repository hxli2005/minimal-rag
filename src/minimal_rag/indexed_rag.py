"""Two tools over a public/read-authorized corpus. The Agent loop stays in loop.py.

SQLite FTS5 supplies BM25, stemming and an inverted index. Chunk hits retain their
exact character offsets; read returns original text. This index must be built for
the caller's authorized corpus; it is not a replacement for an enterprise ACL.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pydantic import Field

from minimal_rag.loop import StrictModel, Tool


class SearchArgs(StrictModel):
    query: str = Field(max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    offset: int = Field(default=0, ge=0)


class ReadArgs(StrictModel):
    document_id: str = Field(min_length=1, max_length=100)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4000, ge=1, le=16000)


class IndexedSearchArgs(SearchArgs):
    scope: str = Field(default="", max_length=250, description="Optional literal substring of a document ID, title, or source path.")
    all_terms: bool = Field(default=False, description="Require every keyword/quoted phrase, rather than matching any of them.")


class IndexedKnowledgeBase:
    def __init__(self, path, excerpt_chars=4000, *, check_same_thread=True, scope_index=None):
        self.path, self.excerpt_chars = Path(path), excerpt_chars
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=check_same_thread)
        self.db.row_factory = sqlite3.Row
        self.scope_index = None
        try:
            self.has_document_ranges = bool(self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_ranges'").fetchone())
            if scope_index is not None and not self.has_document_ranges:
                from minimal_rag.scope_index import ScopeIndex
                self.scope_index = ScopeIndex(scope_index, path)
        except BaseException:
            self.db.close()
            raise

    @staticmethod
    def build(path, documents, chunk_chars=1600, overlap=200):
        """documents yields {id,title,text,source}; never pass question labels here."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not 0 <= overlap < chunk_chars:
            raise ValueError("overlap must be smaller than chunk size")
        path.parent.mkdir(parents=True, exist_ok=True)
        building = path.with_suffix(path.suffix + ".building")
        if building.exists():
            raise FileExistsError(building)
        db = sqlite3.connect(building)
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,text TEXT,source TEXT)")
            db.execute("CREATE TABLE document_ranges(document_id TEXT PRIMARY KEY,first_chunk INTEGER,last_chunk INTEGER)")
            db.execute("CREATE VIRTUAL TABLE chunks USING fts5(document_id UNINDEXED,start UNINDEXED,title,text,tokenize='porter unicode61')")
            n, parts = 0, 0
            for n, doc in enumerate(documents, 1):
                db.execute("INSERT INTO documents VALUES (?,?,?,?)", (doc["id"], doc["title"], doc["text"], doc["source"]))
                text, start, first_chunk = doc["text"], 0, parts + 1
                while start < len(text):
                    end = min(start + chunk_chars, len(text))
                    if end < len(text):
                        boundary = text.rfind("\n", start + chunk_chars // 2, end)
                        if boundary > start:
                            end = boundary + 1
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?)", (doc["id"], start, doc["title"], text[start:end]))
                    parts += 1
                    if end == len(text):
                        break
                    start = max(start + 1, end - overlap)
                db.execute("INSERT INTO document_ranges VALUES (?,?,?)", (doc["id"], first_chunk, parts))
                if n % 1000 == 0:
                    db.commit()
            db.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")
            db.commit()
            stats = {"documents": n, "chunks": parts, "chunk_chars": chunk_chars, "overlap": overlap}
        finally:
            db.close()
        building.replace(path)
        return stats

    def document_bounds(self, document_id):
        if self.has_document_ranges:
            row = self.db.execute("SELECT first_chunk,last_chunk FROM document_ranges WHERE document_id=?",
                                  (document_id,)).fetchone()
            return tuple(row) if row else None
        return self.scope_index.bounds(document_id) if self.scope_index is not None else None

    def search(self, query, limit=5, offset=0, scope="", all_terms=False):
        def window_start(row):
            before = max(0, self.excerpt_chars - len(row["text"])) // 2
            return max(0, min(int(row["start"]) - before, row["document_chars"] - self.excerpt_chars))
        # Treat input as data: only quoted phrases and literal words enter MATCH.
        terms = re.findall(r'"([^"\n]+)"|([\w]+)', query, flags=re.UNICODE)
        terms = list(dict.fromkeys(a or b for a, b in terms))
        match = (" AND " if all_terms else " OR ").join('"' + term.replace('"', '""') + '"' for term in terms)
        pattern = "%" + scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where = "(d.id LIKE ? ESCAPE '\\' OR d.title LIKE ? ESCAPE '\\' OR d.source LIKE ? ESCAPE '\\')"
        in_one_doc = bool(scope and self.db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone())
        if match:
            sql = ("SELECT c.document_id,c.start,c.text,d.title,d.source,length(d.text) AS document_chars,c.rank AS score "
                   "FROM chunks c JOIN documents d ON d.id=c.document_id "
                   "WHERE chunks MATCH ? AND rank MATCH 'bm25(0,0,2,1)'")
            params = [match]
            if in_one_doc:
                sql += " AND c.document_id=?"
                params.append(scope)
                bounds = self.document_bounds(scope)
                if bounds is not None:
                    # Push row bounds into FTS5 before its ranked scan. Keep the
                    # full-corpus BM25 statistics and exact source predicate.
                    sql += " AND c.rowid>=? AND c.rowid<=?"
                    params += list(bounds)
            elif scope:
                sql += " AND " + where
                params += [pattern] * 3
            sql += " ORDER BY rank"
            cursor, rows, seen = self.db.execute(sql, params), [], set()
            try:
                for row in cursor:
                    key = (row["document_id"], window_start(row)) if in_one_doc else row["document_id"]
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(seen) > offset:
                        rows.append(row)
                    if len(rows) > limit:
                        break
            finally:
                cursor.close()
        else:
            sql = ("SELECT d.id AS document_id,0 AS start,d.text,d.title,d.source,length(d.text) AS document_chars,0 AS score "
                   "FROM documents d WHERE " + ("d.id=?" if in_one_doc else where) + " ORDER BY d.id LIMIT ? OFFSET ?")
            rows = self.db.execute(sql, ([scope] if in_one_doc else [pattern] * 3) + [limit + 1, offset]).fetchall()
        hits = []
        for row in rows[:limit]:
            start = window_start(row)
            if self.excerpt_chars > len(row["text"]):
                original = self.db.execute("SELECT text FROM documents WHERE id=?", (row["document_id"],)).fetchone()[0]
                text = original[start:start + self.excerpt_chars]
            else:
                text = row["text"][:self.excerpt_chars]
            hits.append({"document_id": row["document_id"], "title": row["title"], "offset": start,
                         "text": text, "document_chars": row["document_chars"],
                         "truncated": start > 0 or len(text) < row["document_chars"],
                         "source": row["source"], "score": -row["score"]})
        return {"hits": hits, "next_offset": offset + limit if len(rows) > limit else None}

    def read(self, document_id, offset=0, limit=4000):
        row = self.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            return {"error": "not_found_or_not_visible"}
        end = offset + limit
        return {"document_id": row["id"], "title": row["title"], "text": row["text"][offset:end],
                "offset": offset, "document_chars": len(row["text"]), "next_offset": end if end < len(row["text"]) else None,
                "source": row["source"]}

    def close(self):
        self.db.close()
        if self.scope_index is not None:
            self.scope_index.close()


def indexed_tools(kb):
    return {
        "search": Tool("Search original passages with BM25 lexical ranking. Use a complete question or informative terms that preserve "
                       "the distinguishing entities, dates, locations and conditions; overly generic keywords can retrieve a different event. "
                       "Words match any by default. all_terms=true requires every term and can exclude relevant wording; "
                       "quote only phrases expected verbatim. If results are off-topic, vary terminology or relax restrictive terms or scope. "
                       "If results look relevant, limit/offset can expose further candidates instead of repeating similar queries. "
                       "Global results contain distinct documents. scope filters an ID, title or source substring; "
                       "an exact document ID searches passages within that document. "
                       "Hits include original context, offsets and document length; truncated means more text is available through read. "
                       "Empty query lists documents.",
                       IndexedSearchArgs, kb.search),
        "read": Tool("Read original document text by ID. Use a search hit's character offset to read around that passage; "
                     "offset/limit paginate long documents. next_offset indicates more text.", ReadArgs, kb.read),
    }
