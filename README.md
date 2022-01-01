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

If you do not have jupyter notebook, you can download it using pip.

```bash
pip install jupyter
```

Run the jupyter notebook by using the following command on your terminal.

```bash
jupyter notebook
```

## Directory Structure

```
dataset/raw/ -> consist of downloaded csv files from Zillow Prize competition
dataset/processed/ -> consist of processed dataset after running the notebook

notebooks/ -> consist of exploratory data analysis, where our group perform feature engineering to understand the importance of each feature, data preprocessing, and handle missing values

src/models/xgb_model.py -> training XGBoost model
src/models/catboost_model.py -> training CatBoost model
src/models/model_ensemble.py -> ensemble learning (using both XGBoost and CatBoost)

submission/ -> consist of csv file for leaderboard submissions
```

## Getting Started

1. Download or clone this github repository

2. Download the Zillow Prize dataset from Kaggle (can be found on this [link](https://www.kaggle.com/c/zillow-prize-1/data))

3. Place all dataset files in the directory ```dataset/raw/```

4. Run the whole notebook file ```notebooks/explanatory_data_analysis.ipynb``` twice (```CURR_YEAR = 2016``` and ```CURR_YEAR = 2017```)

5. Run ```model_ensemble.py``` by executing following command:
```bash
cd src/models/
python model_ensemble.py
```
You can also run another files: ```xgb_model.py``` and ```catboost_model.py``` by executing similar commands

5. The .csv submission files will be generated inside ```submission/``` folder

## Methodologies
1. First, we need to understand what are the main challenges of this competition. This can be done by summarizing the dataset and see the distributions, number of missing values, etc. Here are the main challenges that our group encountered in this project:
     * The model will very likely to overfit the training data
     * There are a lot of missing values in the dataset

2. After we know the problem, we can quickly do some dirty implementation for the first submission into leaderboard. 

3. And then start addressing the problem stated in step 1. There are a lot of ways to do this, for example overfitting can be addressed by doing feature selection -> removing some redundant features and missing values can be filled with zero or some statistical values like mean, median, and mode. Our group decided to analyze the features one by one, the implementation can be found in ```notebooks/``` folder.

4. Finally, submit the new predictions into leaderboard and see whether the score is improved or not. 

5. Repeat step 3 and 4 until we get the best performance possible.  

## Result

* Public leaderboard: 0.06435 (top-20%)
* Private leaderboard: 0.07514 (top-6%)

## Video

Our project video can be found on this [link](https://youtu.be/Rx7wYivSLV8).
