# Pulsar Star Prediction

A compact machine learning project for classifying pulsar stars from the UCI pulsar dataset using a logistic regression model. The repository is structured as a small MLOps-oriented ML pipeline: it loads and splits data, preprocesses features, trains the model, tracks experiments with MLflow, and exports the model artifact for downstream use.

## Project goal

The objective is to predict whether a signal corresponds to a pulsar star or non-pulsar noise using tabular astrophysics features. The project emphasizes reproducibility, experiment tracking, and artifact management rather than only model accuracy.

## System design and MLOps flow

This project follows a simple but practical ML lifecycle:

1. Data ingestion and validation
   - The dataset is loaded from the raw CSV file.
   - A dedicated Dataset class normalizes column names, checks for the target column, and performs stratified train/validation/test splitting.

2. Feature preprocessing
   - The Preprocessor builds a scikit-learn Pipeline with median imputation.
   - The same fit learned on the training split is applied to validation and test data.
   - Processed feature matrices are saved as CSV files to data/processed.

3. Model training and evaluation
   - The logistic regression model is defined in src/model.py.
   - Training is logged to MLflow using start_run(), log_params(), and log_metrics().
   - Validation and test metrics include accuracy, precision, recall, F1, and ROC-AUC where computable.

4. Experiment tracking and lineage
   - MLflow artifacts are stored under mlruns/.
   - Each run captures parameters, metrics, and the model artifact.
   - This gives a reproducible record of model experiments and comparisons across runs.

5. Model export and artifact packaging
   - export_model.py downloads the model artifact from MLflow and saves a serialized pickle version under saved_models/.
   - A manifest file captures metadata such as run_id, experiment_id, and artifact_path.

6. Data versioning
   - The repository includes DVC configuration under .dvc, with a configured GDrive remote.
   - Raw and processed datasets are tracked in a version-control-friendly workflow, which is a common pattern for ML data governance.

## Repository structure

```text
pulsar-star-prediction/
├── .dvc/                        # DVC metadata and remote config
├── .gitignore
├── .venv/                      # Local virtual environment
├── data/
│   ├── raw/
│   │   └── pulsar_data.csv.dvc
│   └── processed/
│       ├── processed_train.csv.dvc
│       ├── processed_validation.csv.dvc
│       └── processed_test.csv.dvc
├── mlruns/                     # MLflow experiment and run metadata
├── notebooks/
│   └── data_exploration.ipynb
├── saved_models/
│   ├── model.pkl               # Exported model artifact
│   ├── mlflow_run.json         # Export metadata manifest
│   └── model/
├── src/
│   ├── dataset.py              # Data loading and splitting logic
│   ├── model.py                # Logistic regression training and MLflow logging
│   └── preprocess.py           # Imputation-based preprocessing pipeline
├── export_model.py             # Export MLflow model to local artifact directory
├── train.py                    # Main training orchestration script
├── requirements.txt            # Python dependencies
├── LICENSE
├── README.md
└── mlflow.db                   # Local MLflow tracking database
```

## Key implementation details

### Data split design

The model uses a stratified split:

- 20% held out as test data
- 20% of the remaining data reserved as validation data
- final 60% used for training

This pattern helps evaluate generalization while preserving class balance.

### Preprocessing pipeline

The feature pipeline is intentionally simple and production-friendly:

- median imputation for missing values
- no heavy feature engineering in this version
- fits only on the training split to prevent data leakage

### Model training loop

The training flow in train.py does the following:

- loads the dataset
- splits it into train, validation, and test sets
- fits the preprocessor on training features only
- transforms validation and test features
- saves processed CSV files
- instantiates the logistic regression model
- trains it and logs metrics to MLflow

### MLflow integration

MLflow is used for experiment tracking and model artifact storage. The model object logs:

- hyperparameters
- validation metrics
- test metrics
- model artifact under the model path

This allows experiment comparison across runs and reproducible model lineage.

## Local setup

### 1. Clone and create the environment

```powershell
git clone <repository-url>
cd pulsar-star-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Confirm environment and run training

```powershell
python train.py --help
python train.py
```

The default training job reads the raw dataset, saves processed files, and logs a run to MLflow.

## Common commands

### Train a new run

```powershell
python train.py --experiment-name pulsar-star-prediction --run-name logistic-regression-v2
```

### Export a logged run to a local model artifact

```powershell
python export_model.py <RUN_ID> --output-dir saved_models
```

Example:

```powershell
python export_model.py 1234567890abcdef --output-dir saved_models
```

### Inspect MLflow UI locally

From the repository root:

```powershell
mlflow ui
```

Then open the local MLflow UI in a browser, typically at http://localhost:5000.

## Output artifacts

The project produces several useful outputs:

- processed_train.csv
- processed_validation.csv
- processed_test.csv
- MLflow experiment metadata in mlruns/
- local model export and manifest in saved_models/
- MLflow run metadata in mlflow.db

These artifacts are valuable for traceability, auditability, and future deployment or retraining work.

## Tech stack

- Python
- pandas
- scikit-learn
- MLflow
- DVC
- joblib

## Suggested next steps

This project is a strong starting point for a more complete MLOps workflow. Potential next improvements include:

- automated CI/CD checks for training and validation
- model registry promotions and version gating
- parameterized experiment tracking with YAML config files
- Dockerized training environment
- model serving via FastAPI or a lightweight inference API
- monitoring and drift detection for production deployments

## Summary

This repository is a small but realistic ML engineering example: it combines data engineering, preprocessing, model training, experiment tracking, artifact export, and data versioning in a single project. It is especially useful as a foundation for learning how machine learning pipelines are structured in real-world MLOps environments.