import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataProcessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("./data/Data_Adherence_.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["Adherence"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
