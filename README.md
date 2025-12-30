# Experiment Forge - Triple MNIST

Experiment Forge is a scaffold for training deep learning models on the **Triple MNIST** problem using Google Cloud Vertex AI.

## Triple MNIST Problem

The Triple MNIST problem involves:
- **Synthetic Data Generation**: Takes 3 random MNIST images and concatenates them horizontally (28x84 image)
- **Label**: The sum of the three digits (0-27)
- **Model Architecture**: 
  - Splits the 84x28 input into 3 parts (each 28x28)
  - Applies the same neural network to each part to predict 10 logits per digit
  - Applies a convolutional sum layer to combine the 3×10 logits into 28 logits (for sums 0-27)
- **Training**: Uses TensorFlow/Keras with sparse categorical crossentropy loss against one-hot encoded labels (0-27)
- **Visualization**: Uses TensorBoard or other Vertex AI services for monitoring

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Local Training Workflow

```bash
# 1) Download MNIST dataset
expforge bootstrap-mnist --limit 2000

# 2) Generate triple MNIST datasets (concatenates 3 MNIST images)
# Generate training data
expforge generate-triple-mnist-data --count 5000 --output data/synthetic/triple_mnist

# Generate validation data (optional, for model selection)
expforge generate-triple-mnist-data --count 1000 --output data/synthetic/triple_mnist_val

# Generate test data (optional, for final evaluation)
expforge generate-triple-mnist-data --count 1000 --output data/synthetic/triple_mnist_test

# 3) Train the model locally with TensorBoard logging
expforge train-triple-mnist \
    --train data/synthetic/triple_mnist \
    --model-out runs/triple_mnist_model \
    --epochs 10 \
    --batch-size 32 \
    --tensorboard-log-dir runs/tensorboard_logs

# 3b) Continue training from a checkpoint (optional)
# The epoch number is automatically detected - no need to specify --initial-epoch!
expforge train-triple-mnist \
    --train data/synthetic/triple_mnist \
    --model-out runs/triple_mnist_model \
    --epochs 20 \
    --batch-size 32 \
    --resume-from runs/triple_mnist_model.keras \
    --tensorboard-log-dir runs/tensorboard_logs

# 4) Visualize samples
expforge visualize-samples --dataset data/synthetic/triple_mnist --output runs/samples.png
```

### Vertex AI Training

**Initial Setup (one-time):**

1. **Initialize local gcloud:**
   ```bash
   # Run this once to set up the local gcloud installation
   gcloud init
   ```
   This will prompt you to:
   - Log in with your Google account
   - Select or create a configuration
   - Choose your default project (e.g., `lucas-fievet-research`)

2. **Configure Vertex AI settings:**
   ```bash
   expforge config-vertex \
       --project-id lucas-fievet-research \
       --location us-central1 \
       --bucket-name triple-mnist \
       --experiment-name triple-mnist-experiments \
       --tensorboard-name triple-mnist-tensorboard
   ```

**View current configuration:**
```bash
expforge config-vertex --show
```

**Update configuration settings:**
```bash
# Update individual settings
expforge config-vertex --project-id YOUR_PROJECT_ID
expforge config-vertex --location us-central1
expforge config-vertex --bucket-name your-bucket-name
expforge config-vertex --experiment-name your-experiment-name
expforge config-vertex --tensorboard-name your-tensorboard-name
expforge config-vertex --machine-type n1-standard-4

# Or update multiple settings at once
expforge config-vertex \
    --project-id YOUR_PROJECT_ID \
    --location us-central1 \
    --bucket-name your-bucket-name
```

**Check Vertex AI resources:**
```bash
# Check resources (credentials, bucket, TensorBoard, experiments)
expforge check-vertex-resources

# Check with detailed error traces
expforge check-vertex-resources --verbose

# Check and automatically create missing resources (bucket, TensorBoard)
expforge check-vertex-resources --fix
```

**Check Google account authentication:**
```bash
expforge check-account
```
This shows which Google account is currently authenticated and helps verify you're using the correct account for your project.

**Manual setup in GCP Console:**

If you prefer to create resources manually or if automatic creation fails:

1. **Re-authenticate (if token expired):**
   ```bash
   gcloud auth application-default login
   ```

2. **Create GCS Bucket:**
   - Go to [Cloud Storage Console](https://console.cloud.google.com/storage/browser?project=lucas-fievet-research)
   - Click "Create Bucket"
   - Bucket name: `triple-mnist`
   - Location: `us-central1` (must match your config)
   - Storage class: Standard
   - Click "Create"

3. **Create TensorBoard Resource:**
   - Go to [Vertex AI TensorBoard Console](https://console.cloud.google.com/vertex-ai/tensorboards?project=lucas-fievet-research)
   - Click "Create TensorBoard"
   - Display name: `triple-mnist-tensorboard`
   - Region: `us-central1`
   - Click "Create"

4. **Enable Vertex AI API (required for Experiments):**
   ```bash
   gcloud services enable aiplatform.googleapis.com --project=lucas-fievet-research
   ```
   Or via [GCP Console](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=lucas-fievet-research)
   
5. **Experiments:**
   - Experiments are automatically created when you first run training with `--use-vertex`
   - The experiment name will be: `triple-mnist-experiments`
   - No manual creation needed

After creating resources manually, verify with:
```bash
expforge check-vertex-resources
```

**Debugging Credential/Access Issues:**

If you see errors like "invalid_grant: Token has been expired or revoked" or "Error checking bucket":

1. **Check your authenticated account:**
   ```bash
   expforge check-account
   gcloud auth list
   ```

2. **Verify application-default credentials:**
   ```bash
   # Test if credentials work
   gcloud auth application-default print-access-token
   ```
   If this fails, your credentials are expired.

3. **Initialize local gcloud (first time setup):**
   ```bash
   # Run this once to set up the local gcloud installation
   gcloud init
   ```
   This will prompt you to:
   - Log in with your Google account
   - Select or create a configuration
   - Choose your default project

4. **Re-authenticate if needed:**
   ```bash
   # If browser-based auth fails, try without browser (copy/paste URL)
   gcloud auth application-default login --no-browser
   
   # Or use regular gcloud auth first, then application-default
   gcloud auth login
   gcloud auth application-default login
   ```

4. **Verify bucket access:**
   ```bash
   # Test bucket access directly
   gsutil ls gs://triple-mnist/
   ```
   If this works but the check command doesn't, there may be a Python credential cache issue.

**Local training with Vertex AI Experiments:**
```bash
expforge train-triple-mnist \
    --train data/synthetic/triple_mnist \
    --val data/synthetic/triple_mnist_val \
    --test data/synthetic/triple_mnist_test \
    --epochs 10 \
    --batch-size 32 \
    --use-vertex \
    --upload-model
```

**Upload training data to GCS (one-time setup):**

Before submitting a Custom Training Job, you need to upload your data to GCS:

```bash
# Upload training data
gsutil -m cp -r data/synthetic/triple_mnist gs://triple-mnist/data/train

# Upload validation data (if available)
gsutil -m cp -r data/synthetic/triple_mnist_val gs://triple-mnist/data/val

# Upload test data (if available)
gsutil -m cp -r data/synthetic/triple_mnist_test gs://triple-mnist/data/test
```

The `-m` flag enables parallel uploads for faster transfer. The data structure in GCS should match your local structure (with `labels.csv` and `images/` directory).

**Submit Custom Training Job to Vertex AI:**
```bash
expforge train-vertex-job \
    --train-data-gcs gs://triple-mnist/data/train \
    --val-data-gcs gs://triple-mnist/data/val \
    --test-data-gcs gs://triple-mnist/data/test \
    --epochs 10 \
    --batch-size 32 \
    --sync
```

**Train/Validation/Test Set Support:**

✅ **Full support for train, validation, and test sets:**

- **Training**: Required, used for model training
- **Validation**: Optional, used for:
  - Early stopping
  - Model checkpointing (saves best model based on val_accuracy)
  - Hyperparameter tuning
- **Test**: Optional, used for:
  - Final model evaluation
  - Metrics logged to Vertex AI Experiments
  - Reported in CLI output

All three sets are supported in:
- Local training (`train-triple-mnist`)
- Vertex AI Experiments training (`train-triple-mnist --use-vertex`)
- Custom Training Jobs (`train-vertex-job`)

## CLI Commands

The `expforge` CLI provides the following commands. Use `expforge <command> --help` for detailed help on any command.

### `bootstrap-mnist`
Download a compact MNIST subset and save it using the shared dataset format.

**Options:**
- `--output` - Where to store the MNIST subset (default: `data/real/mnist`)
- `--limit` - Number of MNIST samples to download (default: `2000`)
- `--test-split` - Fraction of records to reserve for testing (default: `0.2`)

### `generate-triple-mnist-data`
Generate synthetic triple MNIST dataset by concatenating 3 MNIST images horizontally.

**Note:** To create train/val/test splits, run this command multiple times with different `--output` directories. The `--split` parameter only affects which base MNIST split is used as source data (if your base MNIST has train/test splits).

**Options:**
- `--mnist-dir` - Directory containing base MNIST dataset (default: `data/real/mnist`)
- `--count` - How many triple MNIST samples to generate (default: `5000`)
- `--output` - Directory to store images and labels (default: `data/synthetic/triple_mnist`)
- `--seed` - Seed for reproducibility (default: `42`)
- `--split` - Base MNIST split to use as source: `train` or `test` (default: `train`)

### `train-triple-mnist`
Train the triple MNIST model locally or with Vertex AI Experiments integration.

**Options:**
- `--train` - **Required** - Directory containing training data
- `--val` - Optional directory containing validation data
- `--test` - Optional directory containing test data
- `--model-out` - Where to save the trained model (default: `runs/triple_mnist_model`)
- `--epochs` - Number of training epochs (default: `10`)
- `--batch-size` - Batch size for training (default: `32`)
- `--tensorboard-log-dir` - Directory for TensorBoard logs (local only)
- `--use-vertex` - Use Vertex AI Experiments and Model Registry
- `--upload-model` - Upload model to Vertex AI Model Registry (default: `True`)
- `--resume-from` - Path to a saved model checkpoint to resume training from
- `--initial-epoch` - Starting epoch number when resuming (optional, auto-detected from checkpoint if not specified)

### `visualize-samples`
Generate a grid visualization of triple MNIST samples organized by class (labels 0-27).

**Options:**
- `--dataset` - **Required** - Dataset directory containing `labels.csv` and `images/`
- `--output` - Where to save the grid visualization (default: `runs/samples.png`)
- `--per-class` - How many samples per class to show (default: `4`)

### `config-vertex`
Configure Vertex AI settings (project ID, location, bucket, experiment names, etc.).

**Options:**
- `--project-id` - GCP project ID
- `--location` - GCP region (e.g., `us-central1`)
- `--bucket-name` - GCS bucket name
- `--experiment-name` - Vertex AI Experiment name
- `--tensorboard-name` - TensorBoard resource name
- `--machine-type` - Default machine type for training jobs
- `--show` - Show current configuration

### `train-vertex-job`
Submit a Custom Training Job to Vertex AI for distributed/GPU training.

**Options:**
- `--train-data-gcs` - **Required** - GCS path to training data
- `--val-data-gcs` - Optional GCS path to validation data
- `--test-data-gcs` - Optional GCS path to test data
- `--model-output-gcs` - GCS path for model output
- `--epochs` - Number of training epochs (default: `10`)
- `--batch-size` - Batch size for training (default: `32`)
- `--learning-rate` - Learning rate (default: `0.001`)
- `--machine-type` - Machine type (overrides config)
- `--accelerator-type` - Accelerator type (e.g., `NVIDIA_TESLA_K80`)
- `--accelerator-count` - Number of accelerators (default: `0`)
- `--sync` - Wait for job completion (default: `False`)

## Project Layout

- `src/expforge/cli.py` – Typer CLI for data generation, training, and visualization
- `src/expforge/config.py` – Vertex AI configuration management
- `src/expforge/datasets.py` – Dataset utilities and MNIST bootstrapping
- `src/expforge/synthetic/triple_mnist_generator.py` – Generates triple MNIST samples by concatenating 3 MNIST images
- `src/expforge/training/triple_mnist_model.py` – TensorFlow/Keras model architecture
- `src/expforge/training/vertex_training.py` – Local training functions
- `src/expforge/training/vertex_integration.py` – Vertex AI Experiments integration
- `src/expforge/training/custom_job.py` – Custom Training Job manager
- `src/expforge/training/vertex_train_script.py` – Training script for Vertex AI Custom Jobs
- `src/expforge/visualization/overview.py` – Visualization utilities for triple MNIST (labels 0-27)
- `data/` – Local storage for real/synthetic datasets
- `vertex_config.json` – Vertex AI configuration (created automatically, can be customized)

## Dataset Format

Each dataset directory follows this convention:

```
<dataset>/
  labels.csv         # path,label,split (label is sum 0-27 for triple MNIST)
  images/
    <uuid>.png       # 28x84 grayscale images (3 concatenated MNIST digits)
```

## Model Architecture

The model uses a shared CNN architecture applied to each of the 3 digit regions:

1. **Input**: 28×84×1 grayscale image
2. **Split**: Divides into 3 regions of 28×28 each
3. **Shared CNN**: Applied to each region:
   - Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling
   - Flatten → Dense(128) → Dropout → Dense(10) logits
4. **Combination**: Stacks 3×10 logits and applies Conv1D + Dense layers
5. **Output**: 28 classes (sums from 0 to 27)

## Vertex AI Integration

The project is designed to work with Google Cloud Vertex AI. The following Vertex AI resources are implemented:

### Vertex AI Experiments ✅
- **Experiment tracking**: Compare different hyperparameters and training runs
- **Metrics logging**: Automatic logging of training metrics
- **Parameter tracking**: Log hyperparameters for each run
- **Model linking**: Link trained models to experiment runs
- **Visualization**: Compare runs in Vertex AI Console
- **Location**: `src/expforge/training/vertex_integration.py`
- **Usage**: Use `--use-vertex` flag with `train-triple-mnist`

### Vertex AI Model Registry ✅
- **Model versioning**: Track different model versions
- **Metadata**: Store model metadata (accuracy, architecture, etc.)
- **Deployment**: Easy model deployment to endpoints
- **Lineage**: Track which experiment run produced which model
- **Usage**: Automatically uploads when `--use-vertex` is used

### Vertex AI TensorBoard ✅
- **Real-time metrics**: View training progress in real-time
- **Profiling**: Profile training performance
- **Comparison**: Compare multiple runs side-by-side
- **Integration**: Seamlessly integrated with Experiments
- **Access**: Via Vertex AI Console or TensorBoard UI


### Custom Training Jobs ✅
- **Purpose**: Run training on Vertex AI infrastructure
- **Usage**: For distributed training, GPU training, or managed infrastructure
- **Features**: Automatic code packaging and GCS upload
- **Location**: `src/expforge/training/custom_job.py`

### Cloud Storage (GCS)
- **Data storage**: `gs://triple-mnist/` (configurable)
- **Model artifacts storage**: Automatic upload of trained models
- **Training data staging**: For Custom Training Jobs

**Note**: Vertex AI Datasets and BigQuery are not used in this project. GCS is used directly for data storage, which is simpler for this use case.

## Requirements

- Python 3.9+
- TensorFlow 2.11+
- Google Cloud AI Platform SDK
- See `pyproject.toml` for full dependencies

## GCP Setup

To use Vertex AI features, you'll need:

1. A GCP project with Vertex AI API enabled
2. Authentication configured (via `gcloud auth application-default login` or service account)
3. A GCS bucket for storing data and models
4. (Optional) A Vertex AI TensorBoard instance for experiment tracking

**Configuration:**
The project uses `vertex_config.json` for Vertex AI settings. This file is created automatically with defaults, or you can configure it using:
```bash
expforge config-vertex --show  # View current config
expforge config-vertex --project-id YOUR_PROJECT --location us-central1 --bucket-name YOUR_BUCKET
```

**Default configuration values:**
- **Project ID**: `lucas-fievet-research`
- **Location**: `us-central1`
- **Bucket**: `triple-mnist`
- **Experiment Name**: `triple-mnist-experiments`
- **TensorBoard Name**: `triple-mnist-tensorboard`
- **Machine Type**: `n1-standard-4` (for Custom Training Jobs)

## Code Quality

The codebase has been cleaned up to remove dead code and duplicate functionality:
- All unused functions and methods have been removed
- No duplicate implementations remain
- Imports have been optimized
- Code follows consistent patterns and best practices
