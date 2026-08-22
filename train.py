import argparse

from src.dataset import Dataset
from src.model import LogisticRegressionModel
from src.preprocess import Preprocessor


def run_training(
	data_path="data/raw/pulsar_data.csv",
	output_dir="data/processed",
	experiment_name="pulsar-star-prediction",
	run_name="logistic-regression",
	random_state=42,
):
	"""Run data loading, preprocessing, persistence, training, and evaluation."""
	dataset = Dataset(data_path=data_path, random_state=random_state)
	(
		x_train,
		y_train,
		x_validation,
		y_validation,
		x_test,
		y_test,
	) = dataset.split_data()

	preprocessor = Preprocessor()
	x_train_processed = preprocessor.fit_transform(x_train)
	x_validation_processed = preprocessor.transform(x_validation)
	x_test_processed = preprocessor.transform(x_test)

	processed_files = [
		preprocessor.save_processed_data(
			x_train_processed,
			y_train,
			split_name="train",
			output_dir=output_dir,
			feature_names=x_train.columns,
		),
		preprocessor.save_processed_data(
			x_validation_processed,
			y_validation,
			split_name="validation",
			output_dir=output_dir,
			feature_names=x_validation.columns,
		),
		preprocessor.save_processed_data(
			x_test_processed,
			y_test,
			split_name="test",
			output_dir=output_dir,
			feature_names=x_test.columns,
		),
	]

	model = LogisticRegressionModel(
		experiment_name=experiment_name,
		run_name=run_name,
		max_iter=1000,
		class_weight="balanced",
		random_state=random_state
	)
	model.train(
		x_train_processed,
		y_train,
		validation_features=x_validation_processed,
		validation_target=y_validation,
		test_features=x_test_processed,
		test_target=y_test,
	)

	return model, processed_files


def main():
	parser = argparse.ArgumentParser(
		description="Train and evaluate the pulsar star classifier."
	)
	parser.add_argument(
		"--data-path",
		default="data/raw/pulsar_data.csv",
		help="Path to the raw pulsar CSV file.",
	)
	parser.add_argument(
		"--output-dir",
		default="data/processed",
		help="Directory for processed train, validation, and test CSV files.",
	)
	parser.add_argument(
		"--experiment-name",
		default="pulsar-star-prediction",
		help="MLflow experiment name.",
	)
	parser.add_argument(
		"--run-name",
		default="logistic-regression",
		help="MLflow run name.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Seed used for splitting and model training.",
	)
	args = parser.parse_args()

	model, processed_files = run_training(
		data_path=args.data_path,
		output_dir=args.output_dir,
		experiment_name=args.experiment_name,
		run_name=args.run_name,
		random_state=args.random_state,
	)

	print(f"MLflow run: {model.run_id}")
	print("Processed files:")
	for file_path in processed_files:
		print(f"- {file_path}")


if __name__ == "__main__":
	main()
