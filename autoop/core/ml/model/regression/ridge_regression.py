from autoop.core.ml.model import Model
import numpy as np
from copy import deepcopy


class RidgeRegression(Model):
    """
    Ridge Regression model for linear regression with L2 regularization.

    This model minimizes the least squares error with an added penalty
    proportional to the square of the magnitude of coefficients. The
    regularization term, controlled by alpha, helps prevent overfitting
    by discouraging large coefficients.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initializes the Ridge Regression model."""
        self.alpha = alpha  # Regularization strength
        self._parameters = {'coefficients': None, 'intercept': None}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the ridge regression model to the provided data using
        a closed-form solution.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Raises:
            ValueError: If the input data shapes are incompatible.
        """

        self.validate_fit_and_predict(X, y, is_predict=False)
        # Add a column of ones to observations for the intercept term
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))

        # Closed-form solution of ridge regression:
        # (X.T * X + alpha * I)^(-1) * X.T * y
        n_features = X.shape[1]
        identity = np.eye(n_features)
        identity[0, 0] = 0  # Don't penalize the intercept

        # Ridge closed-form calculation
        inverse_term = np.linalg.inv(X.T @ X + self.alpha * identity)
        self._parameters['coefficients'] = inverse_term @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given feature matrix using
        the trained model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values as a vector.

        Raises:
            ValueError: If the model is not yet fitted or if input
            shapes are incompatible.
        """
        self.validate_fit_and_predict(X, is_predict=True)
        # Add a column of ones to observations for the intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Predict using the linear combination of coefficients and observations
        return X @ self._parameters['coefficients']

    @property
    def get_parameters(self) -> dict:
        """
        Return the copy of learned parameters (intercept and coefficients)
        as a dictionary.
        Returns:
        dict: Dictionary with keys 'intercept' and 'coefficients'
        """
        return deepcopy({
            'intercept': self._parameters[0],
            'coefficients': self._parameters[1:]
        })
