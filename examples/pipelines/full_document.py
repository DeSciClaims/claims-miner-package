def extract(source, ask):
    """One full-document pass with direct finding and evidence extraction."""
    return ask("Read the entire document and extract its findings in one pass.", source)["claims"]
