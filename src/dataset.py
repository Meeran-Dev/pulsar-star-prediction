from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:

    target_column = "target_class"

    def __init__(
        self,
        data_path="data/raw/pulsar_data.csv",
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")

        self.data_path = Path(data_path)
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def load(self):
        data = pd.read_csv(self.data_path)
        data.columns = data.columns.str.strip()

        if self.target_column not in data.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        return data

    def split_data(self):
        data = self.load()
        features = data.drop(columns=self.target_column)
        target = data[self.target_column]

        x_train_validation, x_test, y_train_validation, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

        validation_fraction = self.validation_size / (1 - self.test_size)
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train_validation,
            y_train_validation,
            test_size=validation_fraction,
            random_state=self.random_state,
            stratify=y_train_validation,
        )

        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def split(self):
        return self.split_data()
