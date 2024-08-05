from zenml import pipeline
from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import  model_train
from steps.evaluation import evaluate_model
@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train , X_test, Y_train, Y_test, preprocessor = clean_data(df)
    model = model_train(X_train, Y_train, X_test, Y_test, preprocessor)
    r2_score , mse = evaluate_model(model , X_test,Y_test)
    
    