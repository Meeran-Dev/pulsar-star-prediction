from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_preprocessing_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


class Preprocessor:

    def __init__(self):
        self.pipeline = create_preprocessing_pipeline()

    def fit_impute(self, train_features):
        self.pipeline.fit(train_features)
        return self

    def transform(self, features):
        return self.pipeline.transform(features)

    def fit_transform(self, train_features):
        self.pipeline.fit(train_features)
        return self.pipeline.transform(train_features)

    def save_processed_data(
        self,
        processed_features,
        target=None,
        split_name=None,
        output_dir="data/processed",
        feature_names=None,
    ):
        """Save processed features and optional labels as a CSV file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if split_name:
            filename = f"processed_{split_name}.csv"
        else:
            filename = "processed_data.csv"

        if feature_names is None:
            feature_names = [
                f"feature_{index}"
                for index in range(processed_features.shape[1])
            ]

        processed_data = pd.DataFrame(
            processed_features,
            columns=feature_names,
        )
        if target is not None:
            processed_data["target_class"] = (
                target.to_numpy() if hasattr(target, "to_numpy") else target
            )

        file_path = output_path / filename
        processed_data.to_csv(file_path, index=False)
        return file_path

    def process_and_save(
        self,
        features,
        target=None,
        split_name=None,
        output_dir="data/processed",
        feature_names=None,
    ):
        """Transform features and save the result as a CSV file."""
        processed_features = self.transform(features)
        if feature_names is None and hasattr(features, "columns"):
            feature_names = features.columns

        return self.save_processed_data(
            processed_features,
            target=target,
            split_name=split_name,
            output_dir=output_dir,
            feature_names=feature_names,
        )
