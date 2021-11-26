from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np


def train_xgboost(X_train, X_test, num_ensembles=5):
    # remove outliers
    log_errors = X_train['logerror']
    X_train = X_train[X_train['logerror'] < np.percentile(log_errors, 99.5)]
    X_train = X_train[X_train['logerror'] > np.percentile(log_errors, 0.5)]

    y = X_train['logerror']
    X_train = X_train.drop(['logerror'], axis=1)
    X = X_train.values

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dvalid = xgb.DMatrix(Xvalid, label=yvalid)
    dtest = xgb.DMatrix(X_test.values)

    xgb_params = {'gpu_id':0, # delete if not using gpu
                'tree_method':'gpu_hist', # change to hist if not using gpu
                'min_child_weight': 5, 
                'eta': 0.035, 
                'colsample_bytree': 0.5, 
                'max_depth': 4,
                'subsample': 0.85, 
                'lambda': 0.8,
                'alpha': 0.3,
                'nthread': -1, 
                'booster' : 'gbtree', 
                'silent': 1,
                'seed': 1,
                'gamma' : 0,
                'eval_metric': 'mae', 
                'objective': 'reg:linear'
                }           

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    predictions = np.zeros(X_test.shape[0])
    for i in range(num_ensembles):
        xgb_params['seed'] = i
        model_xgb = xgb.train(xgb_params, dtrain, 1000, watchlist, early_stopping_rounds=100,
                    maximize = False, verbose_eval=100)
        
        predictions += model_xgb.predict(dtest)
    predictions /= num_ensembles

    return predictions


if __name__ == '__main__':
    X_train = pd.read_csv('../../dataset/processed/train_data.csv')
    X_test = pd.read_csv('../../dataset/processed/test_data.csv')
    predictions = train_xgboost(X_train, X_test)

    # submissions 
    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = predictions

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/xgb_prediction.csv', index=False, float_format='%.4f')
    print("Finished writing the file")