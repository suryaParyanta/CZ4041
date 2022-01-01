import pandas as pd
import gc
from catboost_model import train_catboost
from xgb_model import train_xgboost

'''
This file is used to run the model ensemble between catboost and xgboost model. 
'''

if __name__ == '__main__':
    # train on 2016 data
    train_2016 = pd.read_csv('../../dataset/processed/train_data.csv')
    test_2016 = pd.read_csv('../../dataset/processed/test_data.csv')

    print("Training XGBoost Model ...")
    xgb_predictions = train_xgboost(train_2016, test_2016)

    print("Training CatBoost Model ...")
    catboost_predictions = train_catboost(train_2016, test_2016)

    final_predictions_2016 = 0.8 * catboost_predictions + 0.2 * xgb_predictions

    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns:
        if '2016' in c:
            sample_file[c] = final_predictions_2016

    del test_2016
    gc.collect();

    # train on 2017 data
    train_2017 = pd.read_csv('../../dataset/processed/train_data_2017.csv')
    test_2017 = pd.read_csv('../../dataset/processed/test_data_2017.csv')

    final_train = pd.concat([train_2016, train_2017])

    del train_2016, train_2017
    gc.collect();

    print("Training XGBoost Model ...")
    xgb_predictions = train_xgboost(final_train, test_2017)

    print("Training CatBoost Model ...")
    catboost_predictions = train_catboost(final_train, test_2017)

    final_predictions_2017 = 0.8 * catboost_predictions + 0.2 * xgb_predictions
    
    for c in sample_file.columns:
        if '2017' in c:
            sample_file[c] = final_predictions_2017

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/final_ensemble_prediction.csv', index=False, float_format='%.4f')
    print("Finished writing the file")