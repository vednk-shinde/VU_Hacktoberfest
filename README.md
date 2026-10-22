# Privacy-first local enterprise analytics prototype

This repository contains a local deterministic QA engine, a constrained Ollama planner, SQLite query pushdown, dataset version storage, and an experimental multi-step analytics plan executor.

## Setup and local model configuration

The Python implementation uses only the standard library; `requirements.txt` is intentionally empty. Python 3.10 or newer is recommended. The Gemma workflows require Ollama installed separately and the selected model downloaded locally. For example, run `ollama pull gemma3:4b` once and keep the Ollama service running. Deterministic mode does not require Ollama.

The optional local model settings are read from process environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Loopback Ollama HTTP(S) endpoint |
| `OLLAMA_MODEL` | `gemma3:4b` | Installed local model name |
| `OLLAMA_TIMEOUT_SECONDS` | `90` | Per-request timeout |

Copy `.env.example` as a reference if useful; the application does not load `.env` files automatically. Set variables in the shell before running a command. For privacy, the client only accepts loopback Ollama addresses (`localhost`, `127.0.0.1`, or `::1`) and rejects embedded URL credentials, malformed values, non-positive timeouts, and invalid Ollama response shapes. No cloud API or API key is used.

Example PowerShell configuration:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "gemma3:4b"
$env:OLLAMA_TIMEOUT_SECONDS = "90"
```

## Run the synthetic four-dataset demonstration

From the repository root, using its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.demo_enterprise
```

The optional local Gemma planning pass is:

```powershell
.\.venv\Scripts\python.exe -m src.demo_enterprise --gemma
```

The demo uses only synthetic CSVs in `data/enterprise_demo/`. It aggregates two periods, compares units/revenue/average selling price, joins products to inventory and suppliers using declared relationships, and calculates an inventory exposure scenario. If Gemma returns an invalid plan, the validator abstains and the demo runs its deterministic baseline; model success metrics remain false.

Direct file access in `QAEngine` requires an explicit `approved_file_roots` setting. Registered datasets require a `UserContext` that permits the dataset. The plan executor supports only its documented allowlisted operations; it is not a general SQL or code execution interface.

## Dataset validation and profiles

CSV and JSON ingestion preserves source cell values rather than rewriting them. CSV uses UTF-8 (with an optional BOM), rejects empty or duplicate headers, malformed quoting, and rows with extra fields. Short rows are retained with omitted cells treated as missing and reported as a quality warning. A zero-byte or header-only file is represented as an empty dataset; analytical operations that need rows abstain. Numeric, ISO-8601 date, text, boolean, mixed, and empty columns are profiled separately. Mixed and non-finite numeric data cannot be silently aggregated. Profiles report row/column counts, missing values, duplicate rows, empty columns, warnings, and numeric/date summaries. A direct profile includes up to five preview rows; the persistent registry stores the profile without that preview.

Registry imports and replacements require `approved_source_roots`; direct `QAEngine` paths require `approved_file_roots`. Paths are resolved before the root check, including symlinks. The in-memory loader rejects files larger than 64 MiB and datasets larger than 100,000 rows. The multi-step executor also defaults to a 100,000-row dataset/intermediate limit and 40 plan steps. These limits bound this local prototype; larger files require a streaming or external query engine.

## Execution evidence and explanations

Every single- and multi-dataset result includes an additive `evidence.execution` record. It contains an execution ID, status, dataset/version and schema metadata, a content fingerprint, the interpreted intent or validated plan-step trace, executed operations, output row counts, result reference, and profile limitations. Filter values are redacted from the trace. The fixed enterprise workflow also separates recorded source facts, computed output descriptions, and declared assumptions; those descriptions are validated against emitted outputs and plan-produced metrics. Failed validation has an empty operation list; runtime failures are abstentions and are not marked successful. Existing answer and top-level evidence fields remain available.

The enterprise workflow supports regional percentage-change ranking (descending control explicit, nulls last, stable ties), product trend filtering (units increasing while revenue decreases), supplier/inventory joins, and a structured direct/computed/assumption classification. These are workflow-specific capabilities for the four declared datasets, not arbitrary natural-language analytics.

Factual explanations are deterministic summaries of the execution result. The legacy `explain_with_gemma` argument remains accepted for compatibility, but generated model prose is not used as a factual explanation because it cannot be reliably checked against execution evidence. Gemma may propose a plan; validation and deterministic execution remain authoritative.

## Run tests and the deterministic benchmark

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "from src.benchmark import run_benchmark, print_benchmark_report; print_benchmark_report(run_benchmark())"
```

The multi-step planner is an extensible prototype. It has no production-scale benchmark, runtime business graph, feature lineage pipeline, or general-purpose natural-language coverage. Dataset rows are still loaded in memory by the plan executor; the standalone CSV-to-SQLite sync uses bounded insertion batches.

## Evaluate workflow coverage

The live evaluation uses Gemma against unseen questions and a modified synthetic sales file. It records raw proposals, validation/execution status, independent answer matches, latency, and failures without fallback:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_generalization --output outputs\generalization-evaluation.json
```
