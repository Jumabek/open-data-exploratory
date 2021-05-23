"""
regression_cv.py

Regression models
"""

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pingouin as pg
import pandas as pd
import numpy as np
import util
import itertools


def scale_data(train, val, features, target):
    """"

    :param train: pd.DataFrame, training data. Assume column called "data" specifies dataset,
                                "study_id" specifies ID
    :param val: pd.DataFrame, validation data. Assume "study_id" specifies study ID
    :param features: list<str>, the feature set
    :param target: <str>, the target col
    """
    # Drop all NA values
    train = train[features + [target, 'study_id', 'data']].dropna()
    val = val[features + [target, 'study_id', 'data']].dropna()

    # Setup standard scaler
    sc = StandardScaler()
    sc.fit(train[features])

    # Normalize
    train_norm = pd.DataFrame(sc.transform(train[features]), columns=features)
    val_norm = pd.DataFrame(sc.transform(val[features]), columns=features)

    # Add other columns
    train_norm[target] = train[target].values
    train_norm['data'] = train['data'].values
    train_norm['study_id'] = train['study_id'].values
    val_norm[target] = val[target].values
    val_norm['data'] = val['data'].values
    val_norm['study_id'] = val['study_id'].values

    return train_norm, val_norm


def train_validate_model(args):
    """"
    Train an sklearn model and test on validation set

    :param model: sklearn model
    :param train: pd.DataFrame, training data
    :param features: list<str>, the list of features to use
    :param target: str, the name of the column to predict
    :param val: pd.DataFrame, the validation data, leave blank if just training
    :param return_model: <bool>, whether to return the train model

    :return: metric (model, res, true, pred) model, performance if val exists, predicted values
    """
    model, train, model_type, features, target, val, weighted = args
    train, val = scale_data(train=train, val=val, features=features, target=target)

    if weighted:
        # Get sample weights
        weights = util.calculate_sample_weights(train_data=train, val=val, features=features)
        # Train model
        model.fit(X=train[features], y=train[target], sample_weight=weights)
    else:
        # Train model
        model.fit(X=train[features], y=train[target])

    # If val exists validate
    if val is not None:
        y_pred = model.predict(val[features])
        r2 = r2_score(y_true=val[target], y_pred=y_pred)
        mae = mean_absolute_error(y_true=val[target], y_pred=y_pred)
        return model, r2, mae, val[target].values, y_pred, val.study_id.values
    else:
        return model


def chunker(seq, num):
    """[summary]
    Split array into equal sized parts of size num

    :param seq: np.array, the array
    :param num: <int>, the number of parts
    :return: list<array>, a list of arrays of each length
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def get_loso_cv_data(data, features, target):
    """
    Get data to use for cross validation
    Will create folds by days

    :param data: pd.DataFrame, the data, assume has features, target, time_col, 'data', 'study_id' columns
    :param features: list<str>, the features
    :param target: <str>, the target col

    :return: list<pd.DataFrame>, list<pd.DataFrame>, the data to use for the cv (train, val)
    """
    # Drop NA
    data = data[features + [target, 'study_id', 'data']].dropna()

    # Create CV
    train = []
    val = []
    for s in data.study_id.unique():
        # Get all data not in chunk and add to train
        if data.loc[data['study_id'] == s, :].shape[0] >= 30:
            train.append(data.loc[data['study_id'] != s, :])
            val.append(data.loc[data['study_id'] == s, :])

    return train, val


def get_model(model_type, params):
    """
    Get a model with a specific parameter set

    :param model_type: <str>, the type of model
    :param params: <dict>, the model params

    :return: sklearn model, the model with those parameter settings
    """
    # Fill here
    model = Lasso(**params)

    return model


def get_param_combinations(param_dict):
    """
    Get parameter combinations from a dictionary

    :param param_dict: dict<s:list>, the dictionary where parameters are each specified in lists
    :return list<dict>, a list of all the parameter combination dicts
    """
    # Get combinations
    param_combinations = list(itertools.product(*param_dict.values()))
    dict_keys = list(param_dict.keys())

    # Iterate through combinations
    return [dict(zip(dict_keys, v)) for v in param_combinations]


def run_cv(args):
    train, val, data_type, features, targets, time_col, params_dict, \
        models_list, num_folds, weighted = args

    # Lists
    model_type_list = []
    data_list = []
    target_list = []
    fold_list = []
    params_list = []
    r2_list = []
    mae_list = []
    y_pred_list = []
    y_true_list = []
    study_id_list = []

    # For each target
    for target in targets:
        # Model type
        for m in models_list:
            # Get param combinations
            for params in get_param_combinations(params_dict[m]):
                for t, v in zip(*get_loso_cv_data(
                    data=val, features=features, target=target
                )):
                    fold = v.study_id.unique()[0]
                    # If using source data (otherwise keep t which is target != study ID)
                    if data_type == 'cc':
                        t = train.copy()
                    model = get_model(model_type=m, params=params)
                    # Train
                    _, r2, mae, y_true, y_pred, study_id = \
                        train_validate_model(
                            (model, t, m, features, target, v, weighted)
                    )
                    y_true_list.append(str(list(y_true)))
                    y_pred_list.append(str(list(y_pred)))
                    study_id_list.append(str(list(study_id)))
                    print(fold, data_type, target, m, params, np.round(r2, 3), np.round(mae, 3))
                    model_type_list.append(m)
                    data_list.append(data_type)
                    target_list.append(target)
                    fold_list.append(fold)
                    params_list.append(str(params))
                    r2_list.append(r2)
                    mae_list.append(mae)

    # Get df
    res_df = pd.DataFrame({
        'model_type': model_type_list,
        'data': data_list,
        'target': target_list,
        'fold': fold_list,
        'r2': r2_list,
        'mae': mae_list,
        'params': params_list,
        'y_true': y_true_list,
        'y_pred': y_pred_list
    })

    return res_df