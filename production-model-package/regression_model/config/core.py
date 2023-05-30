from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import regression_model

# Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT/ "config.yml"
DATASET_DIR = PACKAGE_ROOT/ "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT/ "trained_models"


class AppConfig(BaseModel):
    """application-level config."""
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """ all configuration relevant to model training and feature engineering. """
    target: float
    features: List[str]
    test_size: float
    random_state: int
    tol: float
    categorical_vars_with_na_frequent: str
    numerical_vars_with_na: int
    temporal_var: int
    dropped_var: int
    numerical_yeo_vars: float
    categorical_var: str