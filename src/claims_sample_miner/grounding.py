"""Recover exact PDF passages despite layout whitespace and Unicode ligatures."""

import unicodedata


def _normalized(text):
    characters, offsets = [], []
    for index, character in enumerate(text):
        if character == "\u00ad":
            continue
        for value in unicodedata.normalize("NFKC", character):
            value = " " if value.isspace() else value
            if value == " " and characters and characters[-1] == " ":
                offsets[-1] = (offsets[-1][0], index + 1)
                continue
            characters.append(value)
            offsets.append((index, index + 1))
    return "".join(characters), offsets


def grounded_passage(spans, span_id, quote):
    """Return a real span and its exact substring; never substitute new evidence.

    An incorrect span ID can be repaired only when the supplied quotation
    identifies one unique source span. Paraphrases and ambiguous matches fail.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None
    needle, _ = _normalized(quote.strip())
    if not needle.strip():
        return None
    matches = []
    for span in spans:
        text = span["text"]
        haystack, positions = _normalized(text)
        start = haystack.find(needle)
        if start < 0:
            continue
        exact = text[positions[start][0]:positions[start + len(needle) - 1][1]]
        match = {"span_id": span["span_id"], "quote": exact}
        if span["span_id"] == span_id:
            return match
        matches.append(match)
    return matches[0] if len(matches) == 1 else None
