def extract(source, ask):
    """Select verbatim result evidence before expressing the scientific findings."""
    evidence = ask(
        'First identify original result passages. Return {"passages": [{"span_id": "...", "quote": "..."}]}; do not yet write claims.',
        source,
    )["passages"]
    spans = {row["span_id"]: row for row in source["spans"]}
    checked = []
    for row in evidence:
        if row["span_id"] not in spans or not row["quote"] or row["quote"] not in spans[row["span_id"]]["text"]:
            raise ValueError("evidence selection returned an invented passage")
        checked.append({"span_id": row["span_id"], "quote": row["quote"], "context": spans[row["span_id"]]["text"]})
    return ask("Express source-faithful findings from these verified passages. Return the claims schema.",
               {"paper": source["paper"], "evidence": checked})["claims"]
