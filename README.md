# Claims sample miner package

An installable extraction package for the weekly competition pilot. It reads a
mounted PDF, extracts page and paragraph spans with the same IDs and 60,000
character extraction budget as `Claims/miner/agent_v1`, asks the permitted
model for a source-grounded ARA artifact, validates the `agent_v1` shape and
references, and retries once with validation feedback. The schema, structural
validator, and output prompt are copied from the canonical Claims miner; the
model loop is a small OpenAI-compatible implementation for Bifrost.

This is a working baseline, not a claim of competitive extraction quality.
Scanned PDFs without text need an OCR-capable reader in a later revision.

## Build and run

Build and install only inside a disposable test environment:

```sh
uv build --wheel --out-dir dist
uv venv /tmp/claims-sample-miner-venv
uv pip install --python /tmp/claims-sample-miner-venv/bin/python dist/*.whl
CLAIMS_INFERENCE_BASE_URL=http://gateway.internal:8080/v1 \
CLAIMS_INFERENCE_TOKEN=<paper-scoped-key> \
/tmp/claims-sample-miner-venv/bin/claims-sample-miner \
  --request examples/request.json \
  --output /tmp/claims-sample-agent-output.json
```

The request is `claims_competition_request_v1`: `paper`, a read-only local
`pdf_path`, and `inference` with a whitelisted `model`, `stream`, and
`max_output_tokens`. Relative PDF paths resolve beside the request file.
The runner captures stdout/stderr and independently validates the output.

## Inference gateway exercise

The worker provides `CLAIMS_INFERENCE_BASE_URL` (ending in `/v1`) and a
paper-scoped `CLAIMS_INFERENCE_TOKEN`. Miner code can read this token; it
never receives the OpenRouter key or Bifrost's administrative key. Bifrost
must run outside the miner execution boundary. The controller accounts for
usage and enforces the published budget.

Run `PYTHONPATH=src python -m unittest discover -s tests` in an environment
with the package dependencies. Tests use a generated real PDF and a mocked
gateway; no model or storage credentials are required.
