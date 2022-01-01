import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def train_catboost(train_data, test_data, num_ensembles : int = 5):
    '''
    Train the data using catboost model

    :param train_data:    Training dataset (in pandas.DataFrame), consists of set of features and ground truth label
    :param test_data:     Test dataset (in pandas.DataFrame), only consists of set of features
    :param num_ensembles: How many catboost model to ensemble
    '''
    y = train_data['logerror']
    train_data = train_data.drop(['logerror'], axis=1)
    train_features = train_data.columns

    # determine features that are considered as categorical variables
    cat_feature_idx = []
    unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = train_data[c].nunique()
        if num_uniques < unique_thresh and not 'sqft' in c and not 'cnt' in c and not 'nbr' in c and not 'number' in c:
            train_data[c] = train_data[c].astype(int)
            test_data[c] = test_data[c].astype(int)
            cat_feature_idx.append(i)

    # train the model
    predictions = np.zeros(test_data.shape[0])
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
        model.fit(train_data, y,
                  cat_features=cat_feature_idx)
        predictions += model.predict(test_data)
    predictions /= num_ensembles

    return predictions


if __name__ == '__main__':
    train_data = pd.read_csv('../../dataset/processed/train_data.csv')
    test_data = pd.read_csv('../../dataset/processed/test_data.csv')
    predictions = train_catboost(train_data, test_data)

    sample_file = pd.read_csv('../../dataset/raw/sample_submission.csv') 
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = predictions

    print('Preparing the csv file ...')
    sample_file.to_csv('../../submission/catboost_prediction.csv', index=False, float_format='%.4f')
    print('Finished writing the file')