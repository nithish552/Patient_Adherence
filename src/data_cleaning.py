import logging
from abc import ABC , abstractmethod

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing_extensions import Annotated

class DataStrategy(ABC):
    """
    Abstract strategy

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod 
    def handle_data(self, data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[np.ndarray, "Y_train"],
    Annotated[np.ndarray, "Y_test"],
    Annotated[ColumnTransformer,"preprocessor"],
] :
        pass 

class DataProcessStrategy(DataStrategy):
    def __init__(self):
        pass
    """"
    Strategy to process data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    'Unnamed: 0.2', 
                    'Unnamed: 0.1', 
                    'Unnamed: 0', 
                    'PatientID'
                ],
                axis=1
            )
            
            numerical_features = data.select_dtypes(include=['number']).columns
            for feature in numerical_features:
                data[feature] = data[feature].fillna(data[feature].mean())

            # For categorical features, fill with the most frequent value
            categorical_features = data.select_dtypes(include=['object']).columns
            for feature in categorical_features:
                data[feature] = data[feature].fillna(data[feature].mode()[0])

            # Convert categorical features to lowercase
            for feature in categorical_features:
                data[feature] = data[feature].str.lower()


            # Remove duplicate rows
            data.drop_duplicates(inplace=True)
            return data
        except Exception as e:
            logging.error(e)
            raise e
class DataDivideStrategy(DataStrategy):
    def __init__(self):
        pass
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[np.ndarray, "Y_train"],
    Annotated[np.ndarray, "Y_test"],
    Annotated[ColumnTransformer,"preprocessor"],
] :
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("Adherence", axis=1)
            y = data["Adherence"]

            # Encode the target variable
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Identify categorical and numerical features
            categorical_features = X.select_dtypes(include=['object']).columns
            numerical_features = X.select_dtypes(exclude=['object']).columns

            # Create transformers for numerical and categorical features
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Handle unseen values
            ])

            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            return X_train, X_test, y_train, y_test, preprocessor
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self):
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)

if __name__ == "__main__":
    data = pd.read_csv('data\Data_Adherence_.csv')
    data_cleaning = DataCleaning(data, DataProcessStrategy())
    data_cleaning.handle_data()
    