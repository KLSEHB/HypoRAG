"""Source excerpts used to validate generated hypothesis anchors."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", re.UNICODE)


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    first = limit * 2 // 3
    return text[:first] + "\n...[middle omitted for request limit]...\n" + text[-(limit - first):]


def exact_source_excerpt(excerpt: str, source: str) -> str | None:
    """Recover a unique contiguous source span after whitespace-only drift."""
    if excerpt in source:
        return excerpt
    needle = [match.group(0) for match in TOKEN_RE.finditer(excerpt)]
    haystack = list(TOKEN_RE.finditer(source))
    if not needle or len(needle) > len(haystack):
        return None
    starts = []
    for position in range(len(haystack) - len(needle) + 1):
        if all(haystack[position + offset].group(0) == token for offset, token in enumerate(needle)):
            starts.append(position)
            if len(starts) > 1:
                return None
    if len(starts) != 1:
        return None
    start = haystack[starts[0]].start()
    end = haystack[starts[0] + len(needle) - 1].end()
    return source[start:end]
