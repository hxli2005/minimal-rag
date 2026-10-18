"""Optional text-corpus tools matching Claude Agent SDK 0.3.301 inputs.

The filesystem contains original text only. Paths are rooted at /corpus; the
SQLite catalog translates paths to source IDs. No query-dependent shortlist.
"""
import asyncio
import heapq
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Literal

from pydantic import Field

from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.loop import StrictModel, Tool

SDK_VERSION = "0.3.301"
MAX_OUTPUT_CHARS = 80000


class GrepArgs(StrictModel):
    pattern: str = Field(min_length=1, description="Regular expression to search for in file contents (ripgrep syntax).")
    path: str = Field(default="/corpus", description="File or directory to search in; defaults to the corpus root.")
    glob: str = Field(default="", description="Glob pattern to filter files, e.g. **/*.txt.")
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches"
    before: int = Field(default=0, ge=0, alias="-B", description="Lines before each match; content mode only.")
    after: int = Field(default=0, ge=0, alias="-A", description="Lines after each match; content mode only.")
    both: int | None = Field(default=None, ge=0, alias="-C", description="Alias for context.")
    context: int | None = Field(default=None, ge=0, description="Lines before and after each match; content mode only.")
    line_numbers: bool = Field(default=True, alias="-n")
    ignore_case: bool = Field(default=False, alias="-i")
    only_matching: bool = Field(default=False, alias="-o", description="Print only matching parts; content mode only.")
    type: str = Field(default="", description="ripgrep file type, e.g. txt; exported documents are text files.")
    head_limit: int = Field(default=250, ge=0, description="Output lines/entries to return; 0 removes this count limit, subject to the output budget.")
    offset: int = Field(default=0, ge=0, description="Skip this many output lines/entries before head_limit.")
    multiline: bool = Field(default=False, description="Allow patterns across lines, including dot matching newlines.")


class GlobArgs(StrictModel):
    pattern: str = Field(min_length=1, description="File glob pattern, e.g. **/*Runbook*.txt.")
    path: str = Field(default="/corpus", description="Directory to search; defaults to the corpus root.")


class FileReadArgs(StrictModel):
    file_path: str = Field(description="Absolute corpus file path returned by search, Grep or Glob.")
    offset: int = Field(default=1, ge=1, description="Starting line number (1-indexed).")
    limit: int | None = Field(default=None, ge=1, description="Number of lines to read; omit to read the file.")
    allow_large: bool = Field(default=False, description="Allow a whole text file over 256 KiB; the tool output budget still applies.")


class CorpusFiles:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.root = self.directory / "text"
        self.catalog = sqlite3.connect((self.directory / "catalog.sqlite").as_uri() + "?mode=ro", uri=True)
        self.rg = shutil.which("rg")
        if not self.rg:
            raise RuntimeError("Grep and Glob require ripgrep (rg)")

    def close(self):
        self.catalog.close()

    def resolve(self, path):
        relative = path.removeprefix("/corpus/") if path.startswith("/corpus/") else ("." if path == "/corpus" else path)
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("path must be inside /corpus")
        if not resolved.exists():
            raise FileNotFoundError(path)
        return resolved

    def file_path(self, document_id):
        row = self.catalog.execute("SELECT path FROM files WHERE id=?", (document_id,)).fetchone()
        return "/corpus/" + row[0] if row else None

    def document_id(self, path):
        relative = path.removeprefix("/corpus/")
        row = self.catalog.execute("SELECT id FROM files WHERE path=?", (relative,)).fetchone()
        return row[0] if row else None

    async def command(self, arguments, path, parse):
        target = self.resolve(path).relative_to(self.root).as_posix()
        # Keep subprocess output off the model context; paginate it below.
        def execute():
            with tempfile.TemporaryFile(mode="w+b") as output:
                result = subprocess.run([self.rg, "--no-config", "--color", "never", *arguments, "--", target],
                                        cwd=self.root, stdout=output, stderr=subprocess.PIPE, timeout=60)
                if result.returncode not in (0, 1):
                    raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
                output.seek(0)
                return parse(line.decode("utf-8").rstrip("\n") for line in output)
        return await asyncio.to_thread(execute)

    async def grep(self, pattern, path="/corpus", glob="", output_mode="files_with_matches", before=0,
                   after=0, both=None, context=None, line_numbers=True, ignore_case=False,
                   only_matching=False, type="", head_limit=250, offset=0, multiline=False):
        # These are authorized text documents, including occasional literal NULs.
        args = ["--text", "--threads", "4", "--with-filename", "--no-heading", "--no-context-separator", "-e", pattern]
        if glob:
            args += ["--glob", glob]
        if type:
            args += ["--type", type]
        if ignore_case:
            args += ["--ignore-case"]
        if multiline:
            args += ["--multiline", "--multiline-dotall"]
        if output_mode == "files_with_matches":
            args += ["--files-with-matches"]
        elif output_mode == "count":
            args += ["--count"]
        else:
            args += ["--line-number" if line_numbers else "--no-line-number", "-B", str(before), "-A", str(after)]
            around = context if context is not None else both
            if around is not None:
                args += ["-C", str(around)]
            if only_matching:
                args += ["--only-matching"]
        # Parallel traversal is much faster on the full corpus. Stable sorting
        # afterwards preserves each file's line order and makes paging repeatable.
        stats = {"total": 0, "matches": 0}
        def parse(stream):
            def observe():
                for line in stream:
                    stats["total"] += 1
                    if output_mode == "count":
                        stats["matches"] += int(line.rsplit(":", 1)[1])
                    yield line
            key = (lambda line: line) if output_mode == "files_with_matches" else self._line_path
            return heapq.nsmallest(offset + min(head_limit or MAX_OUTPUT_CHARS, MAX_OUTPUT_CHARS) + 1, observe(), key=key)
        raw = await self.command(args, path, parse)
        selected = raw[offset:offset + head_limit if head_limit else None]
        lines, used = [], 0
        for line in selected:
            line = line.removeprefix("./")
            line = line if line == "--" else "/corpus/" + line
            if used + len(line) + 1 > MAX_OUTPUT_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        files = list(dict.fromkeys(line if output_mode == "files_with_matches" else self._line_path(line)
                                  for line in lines if line != "--"))
        result = {"mode": output_mode, "numFiles": len(files), "filenames": files,
                  "appliedLimit": head_limit, "appliedOffset": offset,
                  "truncated": offset + len(lines) < stats["total"]}
        if output_mode == "files_with_matches":
            result["totalFiles"] = stats["total"]
        else:
            result.update(content="\n".join(lines), numLines=len(lines), totalLines=stats["total"])
        if output_mode == "count":
            result["numMatches"] = stats["matches"]
        if len(lines) < len(selected):
            result["notice"] = "Output budget reached. Narrow path/pattern or paginate using offset/head_limit."
        elif not lines and stats["total"]:
            result["notice"] = "No entries at this offset; earlier matches exist."
        return result

    @staticmethod
    def _line_path(line):
        # Exported filenames end in .txt and contain neither ':' nor newlines.
        return re.split(r"\.txt(?=[:\-])", line, maxsplit=1)[0] + ".txt"

    async def glob(self, pattern, path="/corpus"):
        if not self.resolve(path).is_dir():
            raise ValueError("Glob.path must be a directory")
        started = time.monotonic()
        # All exported files share the same mtime: no invented source chronology.
        # Equal mtimes are ordered by path. Glob includes hidden/ignored files.
        base = self.resolve(path).relative_to(self.root).as_posix()
        pattern = pattern.removeprefix("/corpus/") if pattern.startswith("/corpus/") else ((base + "/") if base != "." else "") + pattern
        def parse(stream):
            names, count = [], 0
            for count, line in enumerate(stream, 1):
                if count <= 100:
                    names.append("/corpus/" + line.removeprefix("./"))
            return names, count
        names, count = await self.command(["--files", "--hidden", "--no-ignore", "--sort", "path", "--glob", "/" + pattern], path, parse)
        return {"durationMs": round(1000 * (time.monotonic() - started)), "numFiles": len(names),
                "filenames": names, "truncated": count > 100, "totalMatches": count, "countIsComplete": True}

    def read(self, file_path, offset=1, limit=None, allow_large=False):
        path = self.resolve(file_path)
        if not allow_large and offset == 1 and limit is None and path.stat().st_size > 256 * 1024:
            raise ValueError("File exceeds 256 KiB; use offset/limit to read a line range.")
        text = path.read_bytes().decode("utf-8")
        lines = text.splitlines(keepends=True)
        selected = lines[offset - 1:offset - 1 + limit if limit is not None else None]
        content = "".join(selected)
        if len(content) > MAX_OUTPUT_CHARS:
            raise ValueError("Read exceeds the 80000-character output budget; use a smaller line range or Grep.")
        return {"type": "text", "document_id": self.document_id(file_path), "file": {
            "filePath": "/corpus/" + path.relative_to(self.root).as_posix(), "content": content,
            "numLines": len(selected), "startLine": offset, "totalLines": len(lines)},
            **({"notice": "File exists but is empty."} if not lines else
               {"notice": f"Starting line is past the end; file has {len(lines)} lines."} if offset > len(lines) else {})}

    def tools(self, kb):
        tools = indexed_tools(kb)
        original = tools["search"]
        def search(**args):
            result = original.execute(**args)
            for hit in result["hits"]:
                hit["file_path"] = self.file_path(hit["document_id"])
            return result
        tools["search"] = Tool(original.description + " file_path is the same original document available to Grep/Glob/Read.",
                               original.arguments, search)
        provenance = " Only /corpus is accessible. File names contain the source document ID before '--'; cite that ID."
        tools.update({
            "Grep": Tool("Search file contents using ripgrep regular expressions. Choose files_with_matches, content, or count. "
                         "Use path/glob to scope, -i for case insensitivity, -A/-B/-C for context, and head_limit/offset to paginate. "
                         "Invalid patterns return errors. Output is limited to 80000 characters." + provenance, GrepArgs, self.grep),
            "Glob": Tool("Find corpus files by glob pattern, e.g. **/*Runbook*.txt. Returns at most 100 paths; "
                         "narrow the pattern if truncated. Exported files have equal mtimes and ties sort by path." + provenance,
                         GlobArgs, self.glob),
            "Read": Tool("Read an original text file. file_path is a corpus path; offset is the 1-based starting LINE "
                         "and limit is a number of LINES. Omit the range to read the whole file. This text-only tool has "
                         "an 80000-character output budget. The existing read tool instead uses document IDs and character offsets." + provenance,
                         FileReadArgs, self.read),
        })
        return tools

    def evidence_ids(self, name, content):
        """IDs of text actually exposed, for evaluation only; listing != reading."""
        if name == "Read" and content.get("file", {}).get("content"):
            return [content["document_id"]] if content.get("document_id") else []
        if name == "Grep" and content.get("mode") == "content":
            return [doc for path in content["filenames"] if (doc := self.document_id(path))]
        return []
