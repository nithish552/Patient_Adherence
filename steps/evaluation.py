import logging
import mlflow
from typing import Tuple
from typing_extensions import Annotated
import pandas  as pd 
import numpy as np
from zenml import step
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from src.evaluation import MSE , R2, RMSE
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2_score", r2)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating Model: {}".format(e))
        raise e
