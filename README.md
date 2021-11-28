# CZ4041 Project

In this project, we need to predict the log-error between Zestimate and the actual sale price in Fall 2017, based on the features in the Zillow Price dataset (https://www.kaggle.com/c/zillow-prize-1).'

Our group members are as follows: 
1. Aldo Halim
2. Kevin Tarjono
3. Steven Kurnia
4. Surya Paryanta Pattra 

## Installation

Create your own virtual environment and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary library.

```bash
pip install -r requirements.txt
```

## Directory Structure

```
notebooks/ -> consist of exploratory data analysis, where our group perform feature engineering to understand the importance of each feature, and handle missing values

src/models/xgb_model.py -> machine learning models (XGBoost)
src/models/catboost_model.py -> machine learning models (CatBoost)
src/models/model_ensemble.py -> Ensemble Learning (using both XGBoost and CatBoost
```

## Getting Started

1. Download the Zillow Prize dataset from Kaggle (can be found on this [link](https://www.kaggle.com/c/zillow-prize-1/data))
2. Create new directory path ```dataset/raw/``` and place all dataset files in that directory.
3. Run ```notebooks/explanatory_data_analysis.ipynb``` twice (```CURR_YEAR = 2016``` and ```CURR_YEAR = 2017```)
4. Run ```model_ensemble.py``` by executing following command:
```bash
python model_ensemble.py
```
5. The .csv submission file will be generated inside ```submission/```

## Steps Taken
1. To understand the dataset deeply, our group perform feature engineering, and find the importance of each feature from the original dataset, perform some modification (i.e., dataset cleaning, data imputation, additional features). This step is very important throughout the project to find important observation or patterns.

2. Train the models using the pre-processed data
