def extract(source, ask):
    """Extract independently from document sections, then merge by exact statement."""
    groups, current, length = [], [], 0
    for span in source["spans"]:
        if current and length + len(span["text"]) > 20000 and len(groups) < 3:
            groups.append(current)
            current, length = [], 0
        current.append(span)
        length += len(span["text"])
    if current:
        groups.append(current)
    findings = []
    for spans in groups:
        findings.extend(ask(
            "This is one portion of the paper. Extract findings supported here; do not infer missing results.",
            {"paper": source["paper"], "spans": spans},
        )["claims"])
    return findings
