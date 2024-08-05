import logging
from typing import Tuple
from typing_extensions import Annotated
from sklearn.compose import ColumnTransformer
import pandas  as pd 
from zenml import step
import numpy as np 
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataProcessStrategy
@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[np.ndarray, "Y_train"],
    Annotated[np.ndarray, "Y_test"],
    Annotated[ColumnTransformer,"preprocessor"],
] :
    try:
        process_strategy = DataProcessStrategy()
        data_cleaning = DataCleaning(df , process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_clean = DataCleaning(processed_data, divide_strategy)
        X_train,  X_test, Y_train , Y_test, preprocessor = data_clean.handle_data()
        return X_train, X_test, Y_train, Y_test, preprocessor
        logging.info("Data cleaning completed successfully")
    except Exception as e:
        logging.error(f"Error while cleaning data {e}")
        raise e
    