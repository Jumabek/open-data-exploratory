"""
regression_cv.py

Regression models
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import  LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import pingouin as pg
import pandas as pd
import numpy as np
import util
import itertools


def scale_by_participant(train, val, features, target):
    """"

    :param train: pd.DataFrame, training data. Assume column called "data" specifies dataset,
                                "study_id" specifies ID
    :param val: pd.DataFrame, validation data. Assume "study_id" specifies study ID
    :param features: list<str>, the feature set
    :param target: <str>, the target col
    """
    train_dfs = []
    val_dfs = []

    # Get all train/val data study id combinations combinations
    study_ids = pd.concat([train[['data', 'study_id']], val[['data', 'study_id']]]).drop_duplicates()

    for d in study_ids['data'].unique():
        for s in study_ids.loc[study_ids['data'] == d, 'study_id'].unique():
            # Filter
            temp_train = train.loc[(train['data'] == d) & (train['study_id'] == s), :]
            temp_val = val.loc[(val['data'] == d) & (val['study_id'] == s), :]

            # Check if temp_train exists, else use all train to fit scaler
            sc = StandardScaler()
            if temp_train.shape[0] > 0:
                # Fit scaler
                sc.fit(temp_train[features])

                # Create dfs
                train_norm = pd.DataFrame(
                    sc.transform(temp_train[features]), columns=features
                )
                train_norm[target] = temp_train[target].values
                train_norm['data'] = d
                train_norm['study_id'] = s
                train_dfs.append(train_norm)
            else:
                sc.fit(train[features])

            # Check if val data exists (study_id may have dropped out)
            if temp_val.shape[0] > 0:
                val_norm = pd.DataFrame(
                    sc.transform(temp_val[features]), columns=features
                )
                val_norm[target] = temp_val[target].values
                val_norm['data'] = d
                val_norm['study_id'] = s
                val_dfs.append(val_norm)

    # Return concatenated dfs
    return pd.concat(train_dfs).reset_index(drop=True), pd.concat(val_dfs).reset_index(drop=True)


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
    # print(model, target, train.study_id.unique(), val.study_id.unique())
    # train, val = scale_by_participant(train=train, val=val, features=features, target=target)
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
        pearson_corr = pg.corr(x=val[target], y=y_pred, method='pearson').iloc[0]['r']
        pearson_p = pg.corr(x=val[target], y=y_pred, method='pearson').iloc[0]['p-val']
        try:
            skipped_corr = pg.corr(x=val[target], y=y_pred, method='skipped').iloc[0]['r']
            skipped_p = pg.corr(x=val[target], y=y_pred, method='skipped').iloc[0]['p-val']
        except:
            skipped_corr = None
            skipped_p = None
        return model, r2, mae, pearson_corr, pearson_p, skipped_corr, skipped_p, \
            val[target].values, y_pred, val.study_id.values
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


def get_cv_data(data, features, target, time_col='day', num_folds=5):
    """
    Get data to use for cross validation
    Will create folds by days

    :param data: pd.DataFrame, the data, assume has features, target, time_col, 'data', 'study_id' columns
    :param features: list<str>, the features
    :param target: <str>, the target col
    :param time_col: <str>, the column marking time
    :param num_folds: <int>, the number of folds

    :return: list<pd.DataFrame>, list<pd.DataFrame>, the data to use for the cv (train, val)
    """
    # Drop NA
    data = data[features + [target, time_col, 'study_id', 'data']].dropna()

    # Filter out study IDs with < 30 values
    study_ids_keep = data.study_id.value_counts()
    study_ids_keep = study_ids_keep.loc[study_ids_keep >= 30]
    data = data.loc[data.study_id.isin(study_ids_keep), :]

    # Get days sorted
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(by=[time_col]).reset_index(drop=True)

    # Now get the days
    time = data[time_col].unique()

    # Create CV
    train = []
    val = []
    for chunk in chunker(seq=time, num=num_folds):
        # Get all data not in chunk and add to train
        train.append(data.loc[~data[time_col].isin(chunk), :])
        val.append(data.loc[data[time_col].isin(chunk), :])

    return train, val


def get_equal_split_cv_data(data, features, target, time_col='day', num_folds=5):
    """
    Get data to use for cross validation
    Will create folds by days, and splits equally days across folds by participants

    :param data: pd.DataFrame, the data, assume has features, target, time_col, 'data', 'study_id' columns
    :param features: list<str>, the features
    :param target: <str>, the target col
    :param time_col: <str>, the column marking time
    :param num_folds: <int>, the number of folds

    :return: list<pd.DataFrame>, list<pd.DataFrame>, the data to use for the cv (train, val)
    """
    # Drop NA
    data = data[features + [target, time_col, 'study_id', 'data']].dropna()

    # Filter out study IDs with < 30 values
    study_ids_keep = data.study_id.value_counts()
    study_ids_keep = study_ids_keep.loc[study_ids_keep >= 30].index
    data = data.loc[data.study_id.isin(study_ids_keep), :].reset_index(drop=True)

    # Go through each study ID
    train = [[] for i in range(num_folds - 1)]
    val = [[] for i in range(num_folds - 1)]
    for d in data['data'].unique():
        for s in study_ids_keep: # data['study_id'].unique():
            # Get data
            temp = data.loc[(data['study_id'] == s) & (data['data'] == d), :]
            # Check there are enough folds
            if temp.shape[0] > num_folds:
                # Get days sorted
                temp[time_col] = pd.to_datetime(temp[time_col])
                temp = temp.sort_values(by=[time_col]).reset_index(drop=True)

                # Now get the days
                time = temp[time_col].unique()

                # Create CV
                curr = 0
                chunks = chunker(seq=time, num=num_folds)
                for i in range(len(chunks) - 1):
                    # Get all data not in chunk and add to train
                    train[curr].append(temp.loc[temp[time_col] <= chunks[i][-1], :])
                    val[curr].append(temp.loc[temp[time_col].isin(chunks[i + 1]), :])
                    curr += 1
    # Now concatenate
    train = [pd.concat(t).reset_index(drop=True) for t in train]
    val = [pd.concat(v).reset_index(drop=True) for v in val]

    return train, val


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
    if model_type == 'lr':
        model = LinearRegression(**params)
    elif model_type == 'ridge':
        model = Ridge(**params)
    elif model_type == 'lasso':
        model = Lasso(**params)
    elif model_type == 'gbt':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'rf':
        model = RandomForestRegressor(**params)
    elif model_type == 'sv':
        model = SVR(**params)

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
    pearson_corr_list = []
    pearson_p_list = []
    skipped_corr_list = []
    skipped_p_list = []
    weighted_list = []
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
                    # Get model
                    # Go through feature importance list
                    # Concatenate crosscheck data if it exists
                    if data_type == 'both':
                        t = pd.concat([train, t]).reset_index()
                    elif data_type == 'cc':
                        t = train.copy()
                # v = val
                    model = get_model(model_type=m, params=params)
                    # Train
                    _, r2, mae, pearson_corr, pearson_p, skipped_corr, skipped_p, y_true, y_pred, study_id = \
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
                    pearson_corr_list.append(pearson_corr)
                    pearson_p_list.append(pearson_p)
                    skipped_corr_list.append(skipped_corr)
                    skipped_p_list.append(skipped_p)


    # Get df
    res_df = pd.DataFrame({
        'model_type': model_type_list,
        'data': data_list,
        'target': target_list,
        'fold': fold_list,
        'r2': r2_list,
        'mae': mae_list,
        'pearson_corr': pearson_corr_list,
        'pearson_p': pearson_p_list,
        'skipped_corr': skipped_corr_list,
        'skipped_p': skipped_p_list,
        'params': params_list,
        'y_true': y_true_list,
        'y_pred': y_pred_list
    })

    return res_df