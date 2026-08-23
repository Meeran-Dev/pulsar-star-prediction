import argparse
import json
import os
from pathlib import Path

# Local MLflow runs are stored in the file-store layout.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient


def export_model(run_id, output_dir="saved_models", tracking_uri=None):
	"""Save the selected MLflow model as a standalone pickle file."""
	if tracking_uri:
		mlflow.set_tracking_uri(tracking_uri)

	client = MlflowClient()
	run = client.get_run(run_id)
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	mlflow_model_path = Path(
		mlflow.artifacts.download_artifacts(
		run_id=run_id,
		artifact_path="model",
		dst_path=str(output_path),
		tracking_uri=tracking_uri,
		)
	)
	model = mlflow.sklearn.load_model(str(mlflow_model_path))
	model_path = output_path / "model.pkl"
	joblib.dump(model, model_path)

	manifest = {
		"run_id": run_id,
		"experiment_id": run.info.experiment_id,
		"run_name": run.data.tags.get("mlflow.runName"),
		"artifact_path": "model",
		"model_file": model_path.name,
		"format": "pickle",
	}
	(output_path / "mlflow_run.json").write_text(
		json.dumps(manifest, indent=2) + "\n",
		encoding="utf-8",
	)

	return model_path


def main():
	parser = argparse.ArgumentParser(
		description="Export an MLflow run's model artifact for Hugging Face."
	)
	parser.add_argument("run_id", help="MLflow run ID containing the model artifact.")
	parser.add_argument(
		"--output-dir",
		default="saved_models",
		help="Directory where the model artifact and manifest are saved.",
	)
	parser.add_argument(
		"--tracking-uri",
		help="MLflow tracking URI, such as mlruns or http://localhost:5000.",
	)
	args = parser.parse_args()

	model_path = export_model(
		run_id=args.run_id,
		output_dir=args.output_dir,
		tracking_uri=args.tracking_uri,
	)
	print(f"Exported model to: {model_path}")


if __name__ == "__main__":
	main()