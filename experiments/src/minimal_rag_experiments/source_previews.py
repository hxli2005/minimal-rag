"""Optional source discovery without treating unread titles as evidence."""
import inspect

from minimal_rag.loop import Tool


def source_preview_tools(tools, count=10):
    """Keep the requested passages and expose up to `count` extra ranked titles.

    This uses the same authorized search once. It does not change its ranking,
    invent queries, fetch outside the corpus, or add unread text to the ledger.
    Scoped searches and directory listings retain their original behavior.
    """
    if type(count) is not int or not 0 <= count <= 15:
        raise ValueError("source preview count must be between 0 and 15")
    if not count:
        return dict(tools)
    search = tools["search"]

    async def execute(**arguments):
        limit = arguments.get("limit", 5)
        expand = bool(arguments.get("query", "").strip()) and not arguments.get("scope")
        result = search.execute(**{**arguments, "limit": min(20, limit + count) if expand else limit})
        if inspect.isawaitable(result):
            result = await result
        if not expand or "hits" not in result:
            return result
        hits, seen, previews = result["hits"][:limit], set(), []
        seen.update(row["document_id"] for row in hits)
        for row in result["hits"][limit:]:
            if row["document_id"] not in seen:
                previews.append({key: row[key] for key in ("document_id", "title", "source", "offset")})
                seen.add(row["document_id"])
        next_offset = arguments.get("offset", 0) + limit if len(result["hits"]) > limit else result.get("next_offset")
        return {**result, "hits": hits, "next_offset": next_offset, "other_matches": previews}

    return {**tools, "search": Tool(search.description +
            " other_matches contains additional ranked source titles and offsets, without evidence text. "
            "Use read to inspect a promising source before citing it. These pointers do not count as observed evidence.",
            search.arguments, execute)}
