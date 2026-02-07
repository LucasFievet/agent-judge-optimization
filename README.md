# Experiment Forge

Experiment Forge is a scaffold for **nested Markov experiments** (simulator, theory, verifier, scoring) and **Google Cloud Vertex AI** (config, custom training jobs, experiments, model checkpoints).

## Features

- **Simulator**: Generate trajectories from persona/goal configs; optional LLM for messages or fast (no-LLM) mode for verification.
- **Theory**: Compute expected trajectory length, absorption probabilities, and correlations from the same transition model.
- **Verifier**: Check that the simulator matches theory at given sample sizes (confidence intervals).
- **Scoring**: Score samples per experiment, compare two experiments on a metric (e.g. subscribe rate).
- **Vertex AI**: Configure project/bucket/experiment, check resources, submit custom training jobs (placeholder training script included).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**CLI** (from project root):

```bash
# Simulator: run 5 trajectories for experiment "dummy" (writes to base_dir/experiment/dummy/)
expforge simulator run dummy --sample 5

# Without installing: use module runner
PYTHONPATH=src python -m expforge.cli simulator run dummy --sample 5
```

**Experiment output**: By default, results go to **`<project_root>/.data/experiment/<id>/`** (the `.data` folder at the project root; never under `src/`). Set `EXPFORGE_EXPERIMENTS_DIR` or use `--base-dir` / `-d` to override. All commands share the same base directory.

### Vertex AI (optional)

1. Copy `config.json.example` to `config.json` and set your GCP project, location, bucket, experiment name, and TensorBoard name.
2. Authenticate: `gcloud auth application-default login`
3. Check resources: `expforge vertex check-resources` (use `--fix` to create missing bucket/TensorBoard)
4. Submit a custom training job: `expforge vertex train-job --sync`

## CLI commands

- **`expforge verify`** – Run N verification experiments (theory vs simulator), report pass/fail.
- **`expforge simulator run <experiment_id>`** – Generate trajectories; `--no-llm` for fast/verification mode.
- **`expforge verifier run [experiment_id]`** – Single verification run (default: new timestamped experiment).
- **`expforge scoring experiment <experiment_id>`** – Score all samples; write `metrics.yaml`.
- **`expforge scoring compare <id_a> <id_b>`** – Compare two experiments on a metric (e.g. `--metric subscribe`).
- **`expforge vertex config --show`** – Show Vertex config (from `config.json`).
- **`expforge vertex config --project-id ... --location ...`** – Update and save config.
- **`expforge vertex check-resources`** – Check bucket, experiment, TensorBoard; `--fix` to create missing.
- **`expforge vertex train-job`** – Submit Vertex AI custom training job (placeholder script).
- **`expforge check-account`** – Show active gcloud account.

## Configuration

- **Vertex AI**: `config.json` in project root (or `~/.expforge/config.json`). See `config.json.example`. Used for bucket, experiment name, TensorBoard, and training job defaults.
- **Experiments**: All simulator/verifier/scoring outputs use a single base directory; default is `<project_root>/.data`. Override with `EXPFORGE_EXPERIMENTS_DIR` or `--base-dir` / `-d`.

## Project layout

- `src/expforge/cli.py` – Typer CLI (simulator, verifier, scoring, vertex).
- `src/expforge/config.py` – Vertex config load/save; `config.json` path resolution.
- `src/expforge/simulator/` – Experiment simulator (persona, goals, trajectories).
- `src/expforge/theory/` – Markov chain theory (expected length, absorption, correlation).
- `src/expforge/paths.py` – Experiment base path (project root `.data` by default; `EXPFORGE_EXPERIMENTS_DIR` to override).
- `src/expforge/verifier/` – Theory vs simulator verification; experiment load/copy.
- `src/expforge/scoring/` – Experiment scoring and compare.
- `src/expforge/persona/`, `goal/`, `trajectory/` – Persona and goal models; trajectory generation and transition matrix.
- `src/expforge/vertex/` – Vertex AI (bucket, experiment, TensorBoard, run).
- `src/expforge/training/` – Custom job manager and training entrypoint (placeholder) for Vertex.
- `src/expforge/model/` – Checkpoint and deploy utilities (GCS, Vertex Model Registry).
- `src/expforge/data/` – Generic data I/O and GCS helpers (no dataset-specific code).

## Requirements

- Python 3.10+
- See `pyproject.toml` for dependencies (Typer, PyYAML, TensorFlow, Google Cloud AI Platform, etc.).

## Tests

From project root:

```bash
python tests/test_verifier_experiments.py
python tests/test_transitions.py
python tests/test_theory_moments.py
```

Verifier tests look for experiment configs (e.g. `dummy`, `test`, `verifier_1`) under `tests/fixtures/experiments/` or `./experiments/`. Generate with e.g. `expforge simulator run dummy --sample 0 --no-reuse-config -d ./experiments` (use `-d` so outputs are in a known place for tests).
