import logging
import mlflow
from sklearn.compose import ColumnTransformer
import pandas  as pd 
import numpy as np
import mlflow
import sklearn
from zenml import step
from src.model_dev import XGBoostModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from sklearn.pipeline import Pipeline

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_train(
    X_train: pd.DataFrame,
    Y_train: np.ndarray,
    X_test: pd.DataFrame,
    Y_test: np.ndarray,
    preprocessor: ColumnTransformer,
    config : ModelNameConfig,
    )->Pipeline:
    model = None
    if config.model_name == 'xgboost':
        mlflow.sklearn.autolog()
        model = XGBoostModel()
        trained_model = model.train(X_train, Y_train, preprocessor)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name))