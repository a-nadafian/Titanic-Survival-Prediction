import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def objective(trial, X, y):
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.Trial): A trial from the Optuna study.
        X (pd.DataFrame): The training data.
        y (pd.Series): The target variable.

    Returns:
        float: The mean accuracy of the model.
    """
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': 0
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
        scores.append(model.get_best_score()['validation']['Logloss'])

    return np.mean(scores)

def tune_hyperparameters(X, y):
    """
    Tunes the CatBoost hyperparameters using Optuna.

    Args:
        X (pd.DataFrame): The training data.
        y (pd.Series): The target variable.

    Returns:
        dict: The best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    return study.best_params

def train_model(X, y, params):
    """
    Trains the CatBoost model with the given parameters.

    Args:
        X (pd.DataFrame): The training data.
        y (pd.Series): The target variable.
        params (dict): The hyperparameters for the model.

    Returns:
        CatBoostClassifier: The trained model.
    """
    model = CatBoostClassifier(**params, verbose=0)
    model.fit(X, y)
    return model

def generate_submission(model, test_df, passenger_ids):
    """
    Generates the submission file.

    Args:
        model (CatBoostClassifier): The trained model.
        test_df (pd.DataFrame): The test data.
        passenger_ids (pd.Series): The passenger IDs for the submission file.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    preds = model.predict(test_df).astype(int)
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds})
    return submission_df
