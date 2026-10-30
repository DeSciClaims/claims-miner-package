"""Build five distinct controlled extraction packages without editing the sample."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


STRATEGIES = ("full_document", "section_map", "evidence_first", "verify_findings", "results_focus")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-32b")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    people = json.loads(args.participants.read_text())
    if len(people) != len(STRATEGIES):
        raise ValueError("exactly five participants are required")
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for person, strategy in zip(people, STRATEGIES, strict=True):
        name = "claims_test_" + strategy
        destination = args.output / strategy
        if destination.exists():
            raise ValueError(f"refusing to replace existing build: {destination}")
        destination.mkdir()
        with tempfile.TemporaryDirectory(prefix="claims-pipeline-build-") as directory:
            project = Path(directory)
            shutil.copytree(root / "src/claims_sample_miner", project / "src/claims_sample_miner", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            module = project / "src" / name
            module.mkdir()
            (module / "__init__.py").write_text("")
            shutil.copy2(root / "examples/pipelines/runner.py", module / "runner.py")
            shutil.copy2(root / f"examples/pipelines/{strategy}.py", module / "strategy.py")
            config = (root / "pyproject.toml").read_text().replace('name = "claims-sample-miner"', f'name = "{name.replace("_", "-")}"')
            (project / "pyproject.toml").write_text(config)
            subprocess.run(["uv", "build", "--wheel", "--out-dir", str(destination.resolve()), str(project)], check=True)
        wheel, = destination.glob("*.whl")
        manifest = json.loads((root / "examples/competition_manifest.json").read_text())
        manifest.update(package_filename=wheel.name, team_id=person["team_id"], entrypoint=f"{name}.runner:main", models=[args.model])
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        receipts.append({**person, "strategy": strategy, "wheel": str(wheel.resolve()), "manifest": manifest,
                         "package_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()})
    (args.output / "packages.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"Built {len(receipts)} distinct controlled pipelines in {args.output}")


if __name__ == "__main__":
    main()
