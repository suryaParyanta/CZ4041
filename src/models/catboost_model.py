from catboost.core import train
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def train_catboost(X_train, X_test, num_ensembles=5):
    y = X_train['logerror']
    X_train = X_train.drop(['logerror'], axis=1)
    train_features = X_train.columns

    cat_feature_idx = []
    unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = X_train[c].nunique()
        if num_uniques < unique_thresh and not 'sqft' in c and not 'cnt' in c and not 'nbr' in c and not 'number' in c:
            X_train[c] = X_train[c].astype(int)
            X_test[c] = X_test[c].astype(int)
            cat_feature_idx.append(i)

    predictions = 0
    for i in range(num_ensembles):
        model = CatBoostRegressor(iterations=630, 
                                learning_rate=0.033,
                                depth=6, 
                                l2_leaf_reg=3.5,
                                loss_function='MAE',
                                eval_metric='MAE',
                                task_type='GPU',
                                devices='0',
                                random_seed=i)
        model.fit(X_train, y,
                  cat_features=cat_feature_idx)
        predictions += model.predict(X_test)
    predictions /= num_ensembles

    return predictions

if __name__ == '__main__':
    X_train = pd.read_csv('../../dataset/processed/train_data.csv')
    X_test = pd.read_csv('../../dataset/processed/test_data.csv')
    predictions = train_catboost(X_train, X_test)

    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = predictions

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/catboost_prediction.csv', index=False, float_format='%.4f')
    print("Finished writing the file")