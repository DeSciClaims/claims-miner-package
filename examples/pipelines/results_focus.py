import re


def extract(source, ask):
    """Prioritize result-bearing spans using a deterministic lexical retrieval pass."""
    markers = re.compile(r"\b(results?|found|observed|showed|demonstrat\w*|significant\w*|increas\w*|decreas\w*)\b", re.I)
    ranked = sorted(enumerate(source["spans"]), key=lambda item: (
        -len(markers.findall(item[1]["text"])), item[0],
    ))
    chosen, size = [], 0
    for index, span in ranked:
        if size + len(span["text"]) > 40000 and chosen:
            continue
        chosen.append((index, span))
        size += len(span["text"])
    ordered = [span for _, span in sorted(chosen)]
    return ask("Extract the original findings from these result-bearing source passages, preserving qualifications.",
               {"paper": source["paper"], "spans": ordered})["claims"]
