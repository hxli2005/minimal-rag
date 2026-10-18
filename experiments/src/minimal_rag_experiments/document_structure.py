"""Original-text structure and explicit document metadata; no model inference."""
import re


def document_info(text, supplied=None):
    info = {"date": None, "version": None, "superseded": None}
    patterns = {
        "date": r"^(?:last updated|updated(?: at)?|created(?: at)?|document date|recorded)\s*:\s*([^\n]+)",
        "version": r"^(?:document version|version)\s*:\s*([^\n]+)",
        "status": r"^(?:document status|status)\s*:\s*([^\n]+)",
        "superseded_by": r"^(?:superseded by|replaced by)\s*:\s*([^\n]+)",
    }
    evidence = {}
    # Header declarations, not arbitrary dated comments or references to runtime versions.
    for key, pattern in patterns.items():
        match = re.search(pattern, text[:2000], re.I | re.M)
        if match:
            info[key] = match.group(1).strip()
            evidence[key] = {"offset": match.start(), "text": match.group(0)}
    if info.get("superseded_by") or re.match(r"^(superseded|replaced|retired)\b", info.get("status", ""), re.I):
        info["superseded"] = True
    elif re.match(r"^current\b", info.get("status", ""), re.I):
        info["superseded"] = False
    if evidence:
        info["declarations"] = evidence
    if supplied:
        info.update(supplied)
        info["origin"] = "supplied document metadata"
    return info


def sections(text):
    lines = text.splitlines(keepends=True)
    formal, plain, offset, fenced = [], [], 0, False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if re.match(r"^(```|~~~)", line):
            fenced = not fenced
        if not fenced:
            explicit = True
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            confluence = re.match(r"^h([1-6])\.\s+(.+)$", line)
            following = lines[i+1].strip() if i+1 < len(lines) else ""
            if match:
                level, title = len(match.group(1)), match.group(2)
            elif confluence:
                level, title = int(confluence.group(1)), confluence.group(2)
            elif line and re.fullmatch(r"={3,}|-{3,}", following):
                level, title = (1 if following[0] == "=" else 2), line
            else:
                upcoming = [s.strip() for s in lines[i+1:i+3] if s.strip()]
                explicit = False
                list_intro = (line and len(line) <= 120 and (i == 0 or not lines[i-1].strip())
                              and not re.match(r"^(?:[-*•]|\d+[.)])\s", line)
                              and not line.endswith((".", "!", "?"))
                              and any(re.match(r"^(?:[-*•]|\d+[.)])\s", s) for s in upcoming))
                if not list_intro:
                    offset += len(raw)
                    continue
                level, title = 1, line.rstrip(":")
            (formal if explicit else plain).append({"id": f"s{offset}", "title": title, "level": level, "offset": offset, "end": len(text)})
        offset += len(raw)
    nodes = formal or plain
    stack = []
    for node in nodes:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()["end"] = node["offset"]
        stack.append(node)
    return nodes
