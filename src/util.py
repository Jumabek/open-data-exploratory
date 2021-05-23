"""
util.py

Utilities that may be used for multiple purposes.
"""

# Imports
import os
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import shapiro, normaltest
from sklearn.metrics import silhouette_score


def upload_directory(directory, file_type='csv'):
    """
    Upload directory from magma.

    :param directory: <str>, the absolute path to the directory
    :param file_type: <str>, the type of file

    :return: <dict<pd.DataFrame>> the data in a dict where each key is the filename
                                  and each entry is a df of that file
    """
    # Create dict
    files = dict()

    # List file contents
    for f in os.listdir(directory):
        if file_type == 'csv':
            files[i] = pd.read_csv(directory + f, sep=',')
        elif file_type == 'json':
            files[i] = pd.read_json(directory + f, convert_dates=False)

    return files


def percentile(n):
    """
    Percentile outer function
    Call as in "percentile(25)" on another iterable

    :param n: <float>, the percentile to keep

    :return: function, the percentile
    """
    def percentile_(x):
        """
        Percentile inner function

        :param x: <iterable>, some iterable 1D object

        :return: function, the callable function name
        """
        x = x.dropna()
        return np.percentile(x, n, interpolation='nearest')
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def paired_test(df, col1, col2):
    """
    Run a paired test. Assume we are looking at alternative hypothesis
    that df[col1] > df[col2]

    :param df: pd.DataFrame, the df
    :param col1: <str>, the first column
    :param col2: <str>, the second column
    """
    diff = df[col1] - df[col2]
    # Get diff
    s = shapiro(diff)

    # Run test
    if s[1] < 0.05:
        res = pg.wilcoxon(df[col1], df[col2], tail='greater')
    else:
        res = pg.ttest(df[col1], df[col2], tail='greater', paired=True)
        
    return res, s


def non_paired_test(x, y):
    """
    Run a non paired test. Assume we are looking at hypothesis x != y

    :param x: <list>, the first dataset
    :param y: <list>, the second dataset
    """
    all_data = x + y
    # Get diff
    s = normaltest(a=all_data, axis=None)

    # Run test
    if s[1] < 0.05:
        res = pg.mwu(x, y)
    else:
        res = pg.ttest(x, y, paired=False)
        
    return res, s
