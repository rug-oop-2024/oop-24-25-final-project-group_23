
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Attributes: name = name of the feature in the dataset
                feature_type = numerical/categorical
    """

    name: str
    feature_type: Literal["numerical", "categorical"]

    def __str__(self):
        """Returns string of the name and feature type"""
        return f"Feature(name={self.name}, type={self.feature_type})"
