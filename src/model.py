import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


class LogisticRegressionModel:

	def __init__(
		self,
		experiment_name="pulsar-star-prediction",
		run_name=None,
		**model_parameters,
	):
		self.experiment_name = experiment_name
		self.run_name = run_name
		self.model_parameters = model_parameters
		self.model = LogisticRegression(**model_parameters)
		self.run_id = None

	def fit(self, features, target):
		mlflow.set_experiment(self.experiment_name)
		run = mlflow.start_run(run_name=self.run_name)
		self.run_id = run.info.run_id

		try:
			self.model.fit(features, target)
			mlflow.log_params(self.model.get_params())
		except Exception:
			mlflow.end_run(status="FAILED")
			raise

		return self

	def evaluate(self, features, target, dataset_name="validation"):
		predictions = self.model.predict(features)
		probabilities = self.model.predict_proba(features)[:, 1]
		metric_prefix = f"{dataset_name}_"
		metrics = {
			f"{metric_prefix}accuracy": accuracy_score(target, predictions),
			f"{metric_prefix}precision": precision_score(
				target, predictions, zero_division=0
			),
			f"{metric_prefix}recall": recall_score(
				target, predictions, zero_division=0
			),
			f"{metric_prefix}f1": f1_score(target, predictions, zero_division=0),
		}

		try:
			metrics[f"{metric_prefix}roc_auc"] = roc_auc_score(target, probabilities)
		except ValueError:
			pass

		mlflow.log_metrics(metrics)
		return metrics

	def train(
		self,
		train_features,
		train_target,
		validation_features=None,
		validation_target=None,
		test_features=None,
		test_target=None,
	):
		self.fit(train_features, train_target)

		try:
			if validation_features is not None and validation_target is not None:
				self.evaluate(validation_features, validation_target, "validation")
			if test_features is not None and test_target is not None:
				self.evaluate(test_features, test_target, "test")
			mlflow.sklearn.log_model(self.model, "model")
		except Exception:
			mlflow.end_run(status="FAILED")
			raise
		else:
			mlflow.end_run(status="FINISHED")

		return self

	def predict(self, features):
		return self.model.predict(features)

	def predict_proba(self, features):
		return self.model.predict_proba(features)
