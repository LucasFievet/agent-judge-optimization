# Experiment Forge

Experiment Forge is a lightweight scaffold for running repeatable ML/AI experiments. It focuses on:

- Generating synthetic data from pluggable generators (LLM prompts, Markov models, or font-rendered digits).
- Training or prompt-optimizing models against synthetic data first, then real data.
- Visualizing results and keeping runs organized for iteration.
- Preparing for deployment on GCP services like Vertex AI, BigQuery, Firestore, and Cloud Storage.

The repository includes a runnable MNIST example that uses computer fonts to build synthetic images and a simple scikit-learn classifier to show the workflow end to end.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Generate a small MNIST sample, create synthetic digits from computer fonts, then train a baseline classifier:

```bash
# 1) Pull a compact MNIST sample and store it under data/real/mnist
expforge bootstrap-mnist --limit 2000

# 2) Generate 5,000 synthetic images rendered with system fonts
expforge generate-font-digits --count 5000 --output data/synthetic/fonts

# 3) Train a scikit-learn logistic regression on synthetic + real data
expforge train-sklearn --train data/synthetic/fonts --train data/real/mnist --model-out runs/font-baseline.joblib

# 4) Visualize a grid of samples
expforge visualize-samples --dataset data/synthetic/fonts --output runs/font-samples.png
```

## Project layout

- `src/expforge/cli.py` – Typer CLI that wires commands together.
- `src/expforge/synthetic/font_generator.py` – synthetic data generation via computer fonts.
- `src/expforge/training/sklearn_classifier.py` – baseline training loop with scikit-learn.
- `src/expforge/visualization/overview.py` – quick matplotlib utilities.
- `data/` – local storage for real/synthetic datasets; each dataset keeps images and a `labels.csv` manifest.

## Dataset format

Each dataset directory follows the same convention to simplify swapping inputs:

```
<dataset>/
  labels.csv         # path,label,split
  images/
    <uuid>.png
```

The shared CSV and image layout makes it easy to connect new generators or plug in real-world data. Additional metadata columns may be added without breaking loaders.

## Extending

- Add new generators under `src/expforge/synthetic/` that emit the manifest + image layout.
- Add new trainers under `src/expforge/training/` and expose them in `cli.py`.
- Persist run metadata (hyperparameters, metrics) to your GCP backend of choice. The helper functions in `cli.py` keep run configs serializable to JSON for storage in Firestore or Vertex AI Experiments.

## GCP & data platform hooks

The scaffold is designed to be GCP-friendly:

- Buckets: stage `data/` or `runs/` to Cloud Storage for distributed training jobs.
- Vertex AI: package the CLI as a container entrypoint to schedule training/evaluation jobs or prompt-optimization tasks.
- BigQuery/Looker Studio: push `labels.csv` rows and run metrics queries; visualize with Looker dashboards.
- Firestore: store run configs and metrics documents for experiment tracking.
- Cloud Functions/Cloud Run: wrap generator or training steps as callable services.

The code remains cloud-agnostic so you can integrate specific services as you standardize on an orchestrator (Vertex AI Pipelines, Cloud Composer, Dagster, or others).

## MNIST synthetic example

The included MNIST walkthrough uses standard system fonts (DejaVu, Liberation, Noto) to render digits into the MNIST 28x28 format. Real MNIST samples are fetched from OpenML using scikit-learn utilities. Use the `--limit` flag to control how many real samples are downloaded to keep local runs lightweight.
