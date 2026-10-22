# Privacy-first local enterprise analytics prototype

This repository contains a local deterministic QA engine, a constrained Ollama planner, SQLite query pushdown, dataset version storage, and an experimental multi-step analytics plan executor.

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
