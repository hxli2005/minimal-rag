"""Official standard search interface over the project's unchanged FTS5 ranking."""
import json
from pydantic import Field
from minimal_rag.loop import StrictModel, Tool
from shared import DATA


class Query(StrictModel):
    query: str = Field(description='Search query string')


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DATA / 'qwen3-0.6b-tokenizer', local_files_only=True)


def snippet(text, tok):
    tokens = tok.encode(text, add_special_tokens=False)
    return tok.decode(tokens[:512], skip_special_tokens=True) if len(tokens) > 512 else text


def standard_tools(kb, tok):
    def search(query):
        hits = kb.search(query, limit=5)['hits']
        result = []
        for hit in hits:
            docid = hit['document_id']
            text = kb.db.execute('SELECT text FROM documents WHERE id=?', (docid,)).fetchone()[0]
            result.append({'docid': docid, 'snippet': snippet(text, tok)})
        return json.dumps(result, indent=2)
    return {'search': Tool('Perform a search on a knowledge source. Returns top-5 hits with docid, score, '
                           "and snippet. The snippet contains the document's contents (may be truncated based on token limits).",
                           Query, search)}
