import pandas as pd
import gc
from catboost_model import train_catboost
from xgb_model import train_xgboost

if __name__ == '__main__':
    # train 2016
    X_train = pd.read_csv('../../dataset/processed/train_data.csv')
    X_test = pd.read_csv('../../dataset/processed/test_data.csv')

    print("Training XGBoost Model ...")
    xgb_predictions = train_xgboost(X_train, X_test)

    print("Training CatBoost Model ...")
    catboost_predictions = train_catboost(X_train, X_test)

    final_predictions_2016 = 0.8 * catboost_predictions + 0.2 * xgb_predictions

    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns:
        if '2016' in c:
            sample_file[c] = final_predictions_2016

    # free up memory
    del X_test
    gc.collect();

    # train 2017
    X_train_2017 = pd.read_csv('../../dataset/processed/train_data_2017.csv')
    X_test_2017 = pd.read_csv('../../dataset/processed/test_data_2017.csv')

    # Combine train dataset
    final_train = pd.concat([X_train, X_train_2017])

    del X_train_2017, X_train
    gc.collect();

    print("Training XGBoost Model ...")
    xgb_predictions = train_xgboost(final_train, X_test_2017)

    print("Training CatBoost Model ...")
    catboost_predictions = train_catboost(final_train, X_test_2017)

    final_predictions_2017 = 0.8 * catboost_predictions + 0.2 * xgb_predictions
    
    for c in sample_file.columns:
        if '2017' in c:
            sample_file[c] = final_predictions_2017

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/final_ensemble_prediction.csv', index=False, float_format='%.4f')
    print("Finished writing the file")