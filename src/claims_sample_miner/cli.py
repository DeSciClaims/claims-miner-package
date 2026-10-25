"""Extract a structured agent_v1 artifact from one mounted PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pypdf import PdfReader

from .artifact_models import Artifact, Paper
from .artifact_validator import validate_agent_artifact

REQUEST_SCHEMA = "claims_competition_request_v1"
MAX_SOURCE_CHARS = 60_000  # Canonical agent_v1 extraction budget.


def ingest_pdf(pdf_path: Path, paper: dict[str, Any]) -> dict[str, Any]:
    """Use the canonical agent_v1 pypdf span shape and stable span IDs."""
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        raise ValueError("request.pdf_path must name a readable PDF")
    paper_id = str(paper["paper_id"])
    paper = Paper.model_validate(paper).model_dump(mode="json")
    spans: list[dict[str, Any]] = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        page_text = (page.extract_text() or "").strip()
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", page_text) if block.strip()]
        for paragraph_number, paragraph in enumerate(paragraphs, start=1):
            spans.append({
                "span_id": f"{paper_id}-span-{len(spans) + 1:04d}",
                "paper_id": paper_id,
                "section_name": f"Page {page_number}",
                "section_type": "PAGE",
                "page": page_number,
                "char_start": None,
                "char_end": None,
                "text": paragraph,
                "span_type": "text",
                "metadata": {"reader_span_id": f"{paper_id}-p{page_number:03d}-{paragraph_number:03d}", "pdf_reader": "pypdf"},
            })
    if not spans:
        raise ValueError("PDF has no extractable text")
    bounded: list[dict[str, Any]] = []
    remaining = MAX_SOURCE_CHARS
    for span in spans:
        if remaining <= 0:
            break
        text = span["text"][:remaining].rstrip()
        if text:
            bounded.append({**span, "text": text})
            remaining -= len(text)
    return {
        "schema_version": "agent_v1_source_payload_v1",
        "policy_version": "full-source-v1",
        "paper": paper,
        "source_type": "pdf",
        "source_path": str(pdf_path),
        "source_metadata": {"pdf_reader": "pypdf"},
        "spans": bounded,
        "source_integrity": {
            "schema": "claims_source_payload_integrity_v1",
            "truncated": len(bounded) != len(spans) or bounded[-1]["text"] != spans[len(bounded) - 1]["text"],
            "span_count": len(bounded),
            "character_count": sum(len(span["text"]) for span in bounded),
            "page_count": len({span["page"] for span in bounded}),
            "content_sha256": hashlib.sha256("\n\n".join(span["text"] for span in bounded).encode()).hexdigest(),
            "complete_span_count": len(spans),
            "complete_character_count": sum(len(span["text"]) for span in spans),
        },
    }


def completion(*, model: str, messages: list[dict[str, str]], stream: bool, max_tokens: int) -> str:
    """Call Bifrost's OpenAI-compatible worker endpoint with its scoped key."""
    base = os.environ["CLAIMS_INFERENCE_BASE_URL"].rstrip("/")
    token = os.environ["CLAIMS_INFERENCE_TOKEN"]
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0.2,
        "max_tokens": max_tokens, "stream": stream,
    }).encode()
    request = Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1800) as response:
        if not stream:
            return str(json.load(response)["choices"][0]["message"]["content"]).strip()
        chunks: list[str] = []
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError("inference gateway returned a streaming error")
            for choice in event.get("choices", []):
                chunks.append(str(choice.get("delta", {}).get("content") or ""))
        return "".join(chunks).strip()


def _parse_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("model response must be a JSON object")
    return parsed


def _grounding_issues(artifact: Artifact, spans: list[dict[str, Any]]) -> list[str]:
    span_text = {span["span_id"]: span["text"] for span in spans}
    issues: list[str] = []
    for claim in artifact.logic.claims:
        if not claim.sources:
            issues.append(f"claim {claim.claim_id} has no source references")
        for source in claim.sources:
            if not source.span_ids or any(span_id not in span_text for span_id in source.span_ids):
                issues.append(f"claim {claim.claim_id} has unknown source span IDs")
            elif not source.quote or not any(source.quote in span_text[span_id] for span_id in source.span_ids):
                issues.append(f"claim {claim.claim_id} has a quote absent from its source spans")
    return issues


def run(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"request.schema_version must be {REQUEST_SCHEMA}")
    paper = dict(request["paper"])
    paper["paper_id"] = str(paper["paper_id"])
    paper["title"] = str(paper.get("title") or paper["paper_id"])
    pdf_path = Path(request["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = request_path.parent / pdf_path
    source = ingest_pdf(pdf_path, paper)
    inference = request["inference"]
    model = str(inference["model"])
    stream = bool(inference.get("stream", False))
    max_tokens = int(inference.get("max_output_tokens", 8192))
    if not 1 <= max_tokens <= 32768:
        raise ValueError("inference.max_output_tokens is out of range")
    contract = files("claims_sample_miner").joinpath("prompts/agent_v1_contract.md").read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": contract + (
            "\n\nIn this package, paper, source_payload, and agent_schema are embedded "
            "in the user message. There are no file-reading or submit tools. "
            "Return the complete JSON object directly in your reply."
        )},
        {"role": "user", "content": json.dumps({
            "paper": paper, "source_payload": source, "agent_schema": Artifact.model_json_schema(),
        }, ensure_ascii=False)},
    ]
    for attempt in range(2):
        result = completion(model=model, messages=messages, stream=stream, max_tokens=max_tokens)
        try:
            artifact = Artifact.model_validate(_parse_json_object(result))
            if artifact.paper.paper_id != paper["paper_id"]:
                raise ValueError("output paper ID does not match input")
            issues = [item["message"] for item in validate_agent_artifact(artifact)]
            issues.extend(_grounding_issues(artifact, source["spans"]))
            if issues:
                raise ValueError("; ".join(issues))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if attempt:
                raise ValueError(f"agent_v1 output invalid after repair: {exc}") from exc
            messages.extend([
                {"role": "assistant", "content": result},
                {"role": "user", "content": "Repair the full JSON artifact. Validation errors: " + str(exc)},
            ])
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact.paper.title = paper["title"]
        artifact.metadata.update({
            "pipeline_name": "claims_sample_miner",
            "output_schema": "agent_v1",
            "runtime": "bifrost-chat-completions",
            "source_type": "pdf",
            "source_integrity": source["source_integrity"],
        })
        output_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(artifact.logic.claims)} claims for {paper['paper_id']}", file=sys.stderr)
        return
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)


if __name__ == "__main__":
    main()
