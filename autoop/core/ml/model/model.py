from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from pydantic import PrivateAttr


class Model(ABC):
    _parameters = dict = PrivateAttr(default_factory=dict)

    def validate_fit_and_predict(self, X: np.ndarray, y:
                                 np.ndarray, is_predict: bool = True) -> None:
        """ Validates the input arrays for fit and predict methods.

        Args:
            observations (np.ndarray): The observation matrix.
            ground_truth (np.ndarray): The ground truth vector.
            is_fit (bool): Indicates if this is a fit method validation.

        Raises:
            TypeError: If the inputs are not of the expected types.
            ValueError: If the input shapes are not correct.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError("Observations must be an np.ndarray.")

        if not is_predict:
            if not isinstance(y, np.ndarray):
                raise TypeError("Ground_truth must be an np.ndarray")

            if len(X.shape) != 2 or len(y.shape) != 1:
                raise ValueError("Observations must be a 2D array and"
                                 "ground_truth must be a 1D array.")

        else:
            if self._parameters is None:
                raise ValueError("Model has not been trained yet."
                                 "Call fit first.")

            n_features = self._parameters['parameters'].shape[0] - 1
            if X.shape[1] != n_features:
                raise ValueError(f"observations must have {n_features}"
                                 "features (columns)")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Training method blueprint for different linear regression models
        Return: None
        X represents the observations and y represents the ground_truth
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method blueprint for different linear regression models
        Return: ndarray representing a prediction
        """
        pass
