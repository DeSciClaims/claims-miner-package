"""Minimal source-backed baseline for testnet smoke rounds only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_models import (
    Artifact,
    Claim,
    EvidenceLayer,
    EvidenceRecord,
    Logic,
    Paper,
    SourceRef,
    SrcLayer,
    TraceNode,
)
from .artifact_validator import validate_agent_artifact
from .cli import REQUEST_SCHEMA, _grounding_issues, _source_excerpt, ingest_pdf


def run(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("invalid competition request schema")
    paper = Paper.model_validate(request["paper"])
    pdf_path = Path(request["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = request_path.parent / pdf_path
    source = ingest_pdf(pdf_path, paper.model_dump(mode="json"))
    span = next((row for row in source["spans"] if len(row["text"]) >= 60), source["spans"][0])
    quote = _source_excerpt(span["text"], paper.title)
    if not quote or quote not in span["text"]:
        raise ValueError("minimal baseline could not quote the PDF")
    ref = SourceRef(source_id=span["span_id"], source_type="span",
                    span_ids=[span["span_id"]], quote=quote, role="result")
    claim = Claim(
        claim_id="C01", statement=quote,
        conditions="As reported in the cited paper", status="supported",
        falsification_criteria="A contradictory passage in the cited paper",
        evidence_ids=["EV01"], sources=[ref],
    )
    artifact = Artifact(
        paper=paper.model_copy(update={"claims_summary": [claim.statement]}),
        logic=Logic(claims=[claim]),
        evidence=EvidenceLayer(records=[EvidenceRecord(
            evidence_id="EV01", title="First source passage", role="support",
            summary=quote, source_refs=[ref], linked_claim_ids=["C01"],
        )]),
        trace=TraceNode(node_id="root", node_type="extraction",
                        summary="One source-backed passage", source_refs=[ref]),
        src=SrcLayer(environment=["claims-sample-miner-smoke-baseline"]),
        metadata={"pipeline_name": "claims_smoke_minimal_baseline",
                  "output_schema": "agent_v1", "source_type": "pdf",
                  "source_integrity": source["source_integrity"]},
    )
    issues = [item["message"] for item in validate_agent_artifact(artifact)]
    issues.extend(_grounding_issues(artifact, source["spans"]))
    if issues:
        raise ValueError("; ".join(issues))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)


if __name__ == "__main__":
    main()
