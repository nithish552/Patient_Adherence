import logging
from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

class Model(ABC):
    @abstractmethod
    def train(self, X, y, preprocessor):
        pass

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