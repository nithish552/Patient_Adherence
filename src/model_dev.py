import logging
from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    @abstractmethod
    def train(self, X, y, preprocessor):
        pass

class RandomForest(Model):
    def train(self, x_train, y_train,preprocessor, **kwargs):
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(x_train, y_train)
        return pipeline
    
class logisticRegression(Model):
    def train(self, x_train, y_train,preprocessor, **kwargs):
        model = LogisticRegression(max_iter=1000, random_state=42)

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(x_train, y_train)
        return pipeline
    
class GradientBoosting(Model):
    def train(self, x_train, y_train,preprocessor, **kwargs):
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(x_train, y_train)
        return pipeline
    
class MLP_Classifier(Model):
    def train(self, x_train, y_train,preprocessor, **kwargs):
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(x_train, y_train)
        return pipeline

class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train,preprocessor, **kwargs):
        # reg = xgb.XGBRegressor(**kwargs)
        # reg.fit(x_train, y_train)
        model = XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27
        )

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(x_train, y_train)
        return pipeline