"""Rejected document-tool candidate, isolated from the minimal core."""
import json
import sqlite3
from typing import Literal
from pydantic import Field
from minimal_rag.indexed_rag import IndexedKnowledgeBase as BaseKnowledgeBase, ReadArgs, indexed_tools as base_tools
from minimal_rag.loop import Tool

class DocumentReadArgs(ReadArgs):
    view: Literal["text", "toc"] = "text"
    section: str = Field(default="", max_length=40, description="Section ID from toc; offset then counts within that section.")


class IndexedKnowledgeBase(BaseKnowledgeBase):
    def __init__(self, *args, document_tools=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_tools, self._info_cache = document_tools, {}
        self.has_metadata = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='document_metadata'").fetchone())

    @staticmethod
    def build(path, documents, **kwargs):
        metadata = []
        def rows():
            for doc in documents:
                if doc.get('metadata'):
                    metadata.append((doc['id'], json.dumps(doc['metadata'])))
                yield doc
        stats = BaseKnowledgeBase.build(path, rows(), **kwargs)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE document_metadata(document_id TEXT PRIMARY KEY,metadata TEXT)')
            db.executemany('INSERT INTO document_metadata VALUES (?,?)', metadata)
        return stats

    def search(self, *args, **kwargs):
        result = super().search(*args, **kwargs)
        if self.document_tools:
            for hit in result['hits']:
                hit['metadata'] = self.document_details(hit['document_id'])[0]
        return result

    def document_details(self, document_id, text=None):
        if document_id not in self._info_cache:
            from minimal_rag_experiments.document_structure import document_info, sections
            if text is None:
                text = self.db.execute("SELECT text FROM documents WHERE id=?", (document_id,)).fetchone()[0]
            row = self.db.execute("SELECT metadata FROM document_metadata WHERE document_id=?", (document_id,)).fetchone() if self.has_metadata else None
            self._info_cache[document_id] = (document_info(text, json.loads(row[0]) if row else None), sections(text))
        return self._info_cache[document_id]

    def read(self, document_id, offset=0, limit=4000, view="text", section=""):
        row = self.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            return {"error": "not_found_or_not_visible"}
        details = {}
        if self.document_tools:
            metadata, toc = self.document_details(document_id, row["text"])
            details["metadata"] = metadata
            if view == "toc":
                return {"document_id": document_id, "title": row["title"], **details, "sections": toc,
                        "document_chars": len(row["text"]), "source": row["source"]}
            if section:
                node = next((s for s in toc if s["id"] == section), None)
                if node is None:
                    return {"error": "section_not_found", "document_id": document_id}
                start, end = node["offset"] + offset, min(node["offset"] + offset + limit, node["end"])
                return {"document_id": document_id, "title": row["title"], **details, "section": node,
                        "text": row["text"][start:end], "offset": start, "document_chars": len(row["text"]),
                        "next_offset": offset + limit if end < node["end"] else None, "source": row["source"]}
        end = offset + limit
        return {"document_id": row["id"], "title": row["title"], **details, "text": row["text"][offset:end],
                "offset": offset, "document_chars": len(row["text"]), "next_offset": end if end < len(row["text"]) else None,
                "source": row["source"]}

def indexed_tools(kb):
    tools = base_tools(kb)
    if kb.document_tools:
        tools['search'] = Tool(tools['search'].description + " Metadata contains only explicit date/version/supersession declarations; null means unknown.", tools['search'].arguments, kb.search)
        tools['read'] = Tool(tools['read'].description + " view=toc returns original section titles and IDs. section reads that section including children; offset then counts within it. Metadata null means unknown; dated comments are not document dates.", DocumentReadArgs, kb.read)
    return tools
