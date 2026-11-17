"""Training and comparing models."""

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

INT_PARAMS = ['n_estimators', 'max_depth', 'min_child_weight']


def make_pipeline(preprocessor, model) -> Pipeline:
    """Preprocessor + model in one Pipeline.

    The preprocessor is cloned, so every pipeline learns its own scaling / encoding.
    """
    return Pipeline([('prep', clone(preprocessor)), ('model', model)])


def train_val_auc(pipe, X_train, y_train, X_val, y_val):
    """ROC AUC on train and on validation for a fitted model."""
    auc_train = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
    auc_val = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    return auc_train, auc_val


def evaluate_model(name, model, preprocessor, data, params_text=''):
    """Train a pipeline, print ROC AUC and return (fitted pipeline, result row).

    `data` is the tuple (X_train, y_train, X_val, y_val).
    """
    X_train, y_train, X_val, y_val = data
    pipe = make_pipeline(preprocessor, model).fit(X_train, y_train)
    auc_train, auc_val = train_val_auc(pipe, X_train, y_train, X_val, y_val)
    print(f'{name}:  train AUC = {auc_train:.4f}   val AUC = {auc_val:.4f}')
    row = {'Model': name, 'Hyperparameters': params_text,
           'Train ROC AUC': round(auc_train, 4), 'Val ROC AUC': round(auc_val, 4)}
    return pipe, row


def compare_param(make_model, param_name, values, preprocessor, data) -> pd.DataFrame:
    """Train one model for each value of a hyperparameter and show train / val AUC.

    `make_model` is a function that gets one value and returns a new model.
    """
    X_train, y_train, X_val, y_val = data
    rows = []
    for value in values:
        pipe = make_pipeline(preprocessor, make_model(value)).fit(X_train, y_train)
        auc_train, auc_val = train_val_auc(pipe, X_train, y_train, X_val, y_val)
        rows.append({param_name: str(value), 'Train ROC AUC': round(auc_train, 4),
                     'Val ROC AUC': round(auc_val, 4)})
    return pd.DataFrame(rows).set_index(param_name)


def clean_hyperopt_params(params: dict, round_floats: bool = True) -> dict:
    """Hyperopt returns floats for every parameter: make some of them int.

    `round_floats=True` also rounds the other values to 3 digits (nice to read in the table).
    Inside the search we use `round_floats=False`, so hyperopt gets the exact values it suggested.
    """
    return {k: int(v) if k in INT_PARAMS else (round(float(v), 3) if round_floats else float(v))
            for k, v in params.items()}
