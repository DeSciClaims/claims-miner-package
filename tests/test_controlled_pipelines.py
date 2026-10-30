import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from claims_sample_miner.artifact_models import Artifact
from claims_sample_miner.artifact_validator import validate_agent_artifact
from test_package import PAPER_TEXT, write_pdf


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("full_document", "section_map", "evidence_first", "verify_findings", "results_focus")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ControlledPipelinesTests(unittest.TestCase):
    def test_all_strategies_write_grounded_artifacts_and_use_the_expected_calls(self):
        for strategy, expected_calls in zip(NAMES, (1, 1, 2, 2, 1), strict=True):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_pdf(root / "paper.pdf")
                request = root / "request.json"
                request.write_text(json.dumps({
                    "schema_version": "claims_competition_request_v1",
                    "paper": {"paper_id": "example-001", "title": "Example paper"},
                    "pdf_path": "paper.pdf",
                    "inference": {"model": "allowed/model", "allowed_models": ["allowed/model"]},
                }))
                namespace = "controlled_" + strategy
                package = types.ModuleType(namespace)
                package.__path__ = []
                sys.modules[namespace] = package
                load(namespace + ".strategy", ROOT / f"examples/pipelines/{strategy}.py")
                runner = load(namespace + ".runner", ROOT / "examples/pipelines/runner.py")
                claim = {"statement": PAPER_TEXT, "span_id": "example-001-span-0001", "quote": PAPER_TEXT}

                def completion(**kwargs):
                    if 'Return {"passages":' in kwargs["messages"][0]["content"]:
                        return json.dumps({"passages": [{"span_id": claim["span_id"], "quote": PAPER_TEXT}]})
                    return json.dumps({"claims": [claim]})

                output = root / "output.json"
                with patch.object(runner, "completion", side_effect=completion), patch.object(sys, "argv", ["pipeline", "--request", str(request), "--output", str(output)]):
                    runner.main()
                artifact = Artifact.model_validate_json(output.read_text())
                self.assertEqual(validate_agent_artifact(artifact), [])
                self.assertEqual(artifact.metadata["model_calls"], expected_calls)
                self.assertEqual(artifact.logic.claims[0].sources[0].quote, PAPER_TEXT)

    def test_evidence_first_rejects_fabricated_passages_before_the_second_call(self):
        module = load("controlled_evidence_rejection", ROOT / "examples/pipelines/evidence_first.py")
        calls = []

        def ask(*args):
            calls.append(args)
            return {"passages": [{"span_id": "real", "quote": "invented"}]}

        with self.assertRaisesRegex(ValueError, "invented passage"):
            module.extract({"spans": [{"span_id": "real", "text": PAPER_TEXT}]}, ask)
        self.assertEqual(len(calls), 1)
