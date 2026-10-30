def extract(source, ask):
    """Draft findings, then independently check scope, novelty and support."""
    draft = ask("Draft the original scientific findings with exact evidence.", source)["claims"]
    return ask(
        "Audit these candidate findings against the complete source. Remove background, unsupported or overstated claims; correct scope and evidence. Return only the corrected claims schema.",
        {"source": source, "draft_findings": draft},
    )["claims"]
