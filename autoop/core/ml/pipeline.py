from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ) -> None:
        """Pipeline initialization"""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical" and
           model.type != "classification"):
            raise ValueError("Model type must be classification for "
                             "categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous "
                             "target feature")

    def __str__(self) -> str:
        """String repr of pipeline"""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Getter for the model"""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the pipeline
           execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_"
                                                 f"{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """Register artifact"""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess features"""
        (target_feature_name, target_data, artifact) = preprocess_features(
                                                       [self._target_feature],
                                                       self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in
                               input_results]

    def _split_data(self) -> None:
        """Split the data into training and testing sets"""
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in
                         self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in
                        self._input_vectors]
        self._train_y = self._output_vector[:int(split *
                                                 len(self._output_vector))]
        self._test_y = self._output_vector[int(split *
                                               len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Compact vectors
        Returns: np.array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train model"""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluate using metrics"""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """Execution of the pipeline
        Returns: Dict with metrics and predictions
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        # Evaluate on the training set
        self._test_X, self._test_y = self._train_X, self._train_y
        self._evaluate()  # Evaluate on training data
        train_metrics_results = [(f"train_{metric}", result) for metric,
                                 result in self._metrics_results]
        train_predictions = self._predictions

        # Evaluate on the test set
        self._test_X, self._test_y = self._test_X, self._test_y
        self._evaluate()  # Evaluate on test data
        test_metrics_results = [(f"test_{metric}", result) for metric,
                                result in self._metrics_results]
        test_predictions = self._predictions

        # Combine metrics for both datasets
        all_metrics_results = train_metrics_results + test_metrics_results

        return {
            "metrics": all_metrics_results,
            "predictions": {
                "train": train_predictions,
                "test": test_predictions
            }
        }
