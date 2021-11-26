import pandas as pd
import numpy as np
from catboost_model import train_catboost
from xgb_model import train_xgboost

if __name__ == '__main__':
    X_train = pd.read_csv('../../dataset/processed/train_data.csv')
    X_test = pd.read_csv('../../dataset/processed/test_data.csv')

    print("Training XGBoost Model ...")
    xgb_predictions = train_xgboost(X_train, X_test)

    print("Training CatBoost Model ...")
    catboost_predictions = train_catboost(X_train, X_test)

    final_predictions = 0.8 * catboost_predictions + 0.2 * xgb_predictions

    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = final_predictions

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/ensemble_prediction.csv', index=False, float_format='%.4f')
    print("Finished writing the file")