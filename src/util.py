"""
util.py

Utilities that may be used for multiple purposes.
"""

# Imports
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import pandas as pd
import numpy as np
import io
import pingouin as pg
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree as KDTree
from scipy.stats import shapiro, wilcoxon, ttest_rel, normaltest
from sklearn.metrics import silhouette_score
import math
import ot


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


def reduce_correlated_vars(df, features, outcomes, vif_cut=5):
    """[summary]
    Dimensions reduction using the variance inflation factor (VIF)
    Orders by avg correlation between features and endogenous vars

    :param df: pd.DataFrame, the dataframe w/ features and outcomers
    :param features: list<str>, list of features (exogenous vars)
    :param outcomes: list<str>, list of outcomes (endogenous vars)
    :param vif_cut: <int>, the cutoff for the VIF

    :return: [description]
    """

    # First get values correlated to the outcomes
    df_corr = df[features + outcomes].corr().loc[features, outcomes]

    # Calculate average correlation and sort
    df_corr['avg'] = df_corr[outcomes].mean(axis=1)
    df_corr.sort_values(by='avg', inplace=True)

    # Add values
    keep_features = []
    for i in df_corr.index:
        if len(keep_features) > 0:
            vif = variance_inflation_factor(
                df[keep_features + [i]].values, 
                len(keep_features)
            )
            if vif < vif_cut:
                keep_features += [i]
        else:
            keep_features += [i]
    # Check if values changes
    return keep_features


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


def mcc(y_true, y_pred):
    """
    Compute the Matthew's Correlation Coefficient

    :param y_true: <np.array>, the true y value
    :param y_pred: <np.array>, the predicted loss

    :return: <float>, the MCC 
    """
    
    return matthews_corrcoef(y_true, y_pred)


def acc(tp, tn, fp, fn):
    """
    Calculate accuracy

    :param tp: <int>, Number of true positives
    :param tn: <int>, Number of true negatives
    :param fp: <int>, Number of false positives
    :param fn: <int>, Number of false negatives
    """
    return (tp + tn) / (tp + tn + fn + fp)


def tpr(tp, fn):
    """
    Calculate true positive rate (recall)

    :param tp: <int>, Number of true positives
    :param fn: <int>, Number of false negatives
    """
    return tp / (tp + fn)


def tnr(tn, fp):
    """
    Calculate true negative rate

    :param tn: <int>, Number of true negatives
    :param fp: <int>, Number of false positives
    """
    return tn / (tn + fp)


def ppv(tp, fp):
    """
    Calculate the positive predictive value (precision)

    :param tp: <int>, Number of true positives
    :param fp: <int>, Number of false positives
    """
    return tp / (tp + fp)


def f1(tp, fp, fn):
    """
    Calculate f1 score

    :param tp: <int>, Number of true positives
    :param fp: <int>, Number of false positives
    :param fn: <int>, Number of false negatives
    """
    tpr_val = tpr(tp, fn)
    ppv_val = ppv(tp, fp)

    return 2 * (tpr_val * ppv_val) / (tpr_val + ppv_val)

def bacc(tp, tn, fp, fn):
    """
    Calculate balanced accuracy

    :param tp: <int>, Number of true positives
    :param tn: <int>, Number of true negatives
    :param fp: <int>, Number of false positives
    :param fn: <int>, Number of false negatives
    """
    tpr_val = tpr(tp, fn)
    tnr_val = tnr(tn, fp)

    return (tpr_val + tnr_val) / 2


def mcc2(tp, tn, fp, fn):
    """
    Calculate MCC from the confusion matric

    :param tp: <int>, Number of true positives
    :param tn: <int>, Number of true negatives
    :param fp: <int>, Number of false positives
    :param fn: <int>, Number of false negatives
    """
    return ((tp * tn) - (fp * fn)) / \
        np.sqrt(
            (tp + fp) * (tp + fn) * 
            (tn + fp) * (tn + fn)
        )


def js(x, y):
    """
    :param x: np.array, Samples from distribution P, which typically represents the true
        distribution.
    :param y: np.array, Samples from distribution Q, which typically represents the approximate
        distribution.
    
    :return: <float> , The estimated Kullback-Leibler divergence D(P||Q).
    """
    return 0.5 * (kl(x, y) + kl(y, x))


def kl(x, y, center=False):
    """
    Compute the Kullback-Leibler divergence between two multivariate samples.
    From: https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Reference: Pérez-Cruz, F. Kullback-Leibler divergence estimation of 
            continuous distributions IEEE International Symposium on Information
            Theory, 2008.

    :param x: pd.DataFrame, Samples from distribution P, which typically represents the true
    distribution.
    :param y: pd.DataFrame, Samples from distribution Q, which typically represents the approximate
    distribution.
    :param center: <bool>, whetehr to only focus on the center of the data

    :return: <float> , The estimated Kullback-Leibler divergence D(P||Q).
    """
    # Filter if center
    if center:
        x = x.loc[x.abs().max(axis=1) < 5, :]
        y = y.loc[y.abs().max(axis=1) < 5, :]

    # Check if data points remain
    if (x.shape[0] == 0) or (y.shape[0] == 0):
        return None

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    try:
        r_temp = xtree.query(x, k=x.shape[0])[0]
    except ValueError:
        return None
    s = ytree.query(x, k=1)[0]

    r = []
    for i in range(r_temp.shape[0]):
        if len(r_temp.shape) > 1:
            for j in range(r_temp.shape[1]):
                if r_temp[i, j] > 0:
                    r.append(r_temp[i, j])
                    break
        else:
            return None
    r = np.array(r)

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def calculate_emd(d1, d2, center=False):
    """
    Calculate earth mover's distance

    :param d1: np.array, first data
    :param d2: np.array, second data
    :param center: <bool>, whether to only use the center (5 std dev)
    :return: <float>, the EMD
    """
    # Filter if center
    if center:
        d1 = d1.loc[d1.abs().max(axis=1) < 5, :]
        d2 = d2.loc[d2.abs().max(axis=1) < 5, :]

    if (d1.shape[0] == 0) or (d2.shape[0] == 0):
        return None
    a, b = np.ones((d1.shape[0],)) / d1.shape[0], np.ones((d2.shape[0],)) / d2.shape[0]
    M = ot.dist(d1.values, d2.values, metric='euclidean')
    # return ot.emd2(a=a, b=b, M=M)
    return ot.sinkhorn2(a=a, b=b, M=M, reg=1)[0]


def mv_skew(x):
    """
    Multivariate skew from Mardia's measure

    :param x: np.array, 2D array of data
                        rows are samples, cols are features

    :return: <float>, the multivariate skew
    """
    # Calculate metrics
    x_bar = np.mean(x, axis=0)
    cov = np.cov(np.transpose(x), bias=True)
    cov_inv = np.linalg.pinv(cov)

    # Now calculate skew
    x_minus_x_bar = x - x_bar
    total = np.matmul(x_minus_x_bar, np.matmul(cov_inv, np.transpose(x_minus_x_bar)))**3
    
    return np.sum(total) / (x.shape[0]**2)


def mv_kurtosis(x):
    """
    Multivariate kurtosis from Mardia's measure

    :param x: np.array, 2D array of data
                        rows are samples, cols are features

    :return: <float>, the multivariate skew
    """
    # Calculate metrics
    x_bar = np.mean(x, axis=0)
    cov = np.cov(np.transpose(x), bias=True)
    cov_inv = np.linalg.pinv(cov)

    # Now calculate skew
    x_minus_x_bar = x - x_bar
    total = np.matmul(x_minus_x_bar, np.matmul(cov_inv, np.transpose(x_minus_x_bar)))**2
    
    return np.sum(np.diagonal(total)) / x.shape[0]


class PreClustering(object):
    """
    KMeans Clustering object to use for MTL
    """

    def __init__(self, model_type='kmeans'):
        """
        Initilize the kmeans clustering object

        :param model_type: <str>, whether to use kmeans or not
        """
        self.model_type = model_type
        self.model = None
        self.k = None
        self.silhouette_score = 0
        self.pca = None
        self.sc = None

    def train(self, data, features, klow=2, khigh=10):
        """
        Train the kmeans object

        :param data: pd.DataFrame, the data from bl
        :param features: list, the features for the model
        :param klow: <int> the lowest number of clusters to test
        :param khigh: <int>, the highest number of clusters to test
        """
        if self.model_type == 'intervention':
            self.k = 2
            print('Silhouette = ', 'N/A', 'K = ', self.k)
            return None, 2
        self.sc = StandardScaler()
        self.sc.fit(data[features])
        data_scaled = self.sc.transform(data[features])

        print('Running clustering...')
        num_components = []
        k = []
        clustering_type = []
        silhouette_scores = []
        explained_variance = []
        exp_var = 0
        n = 2
        # while exp_var < 0.99:  # Run until 99% of the variance explained
        #     pca = PCA(n_components=n)
        #     pca.fit(data_scaled)
        #     data_pca = pca.transform(data_scaled)
        #     exp_var = np.sum(pca.explained_variance_ratio_)
        for i in range(klow, khigh + 1):
            for c in [
                KMeans(n_clusters=i, random_state=np.random.randint(low=0, high=100)), 
                AgglomerativeClustering(n_clusters=i)
            ]:
                curr_score = []
                for _ in range(10):
                    c.fit(data_scaled)
                    curr_score.append(silhouette_score(data_scaled, c.labels_))
                print(n, i, type(c).__name__, np.mean(curr_score))
                num_components.append(n)
                k.append(i)
                clustering_type.append(type(c).__name__)
                silhouette_scores.append(np.mean(curr_score))
                explained_variance.append(exp_var)
        n += 1

        # Put in dataframe and save
        cluster_df = pd.DataFrame({
            'N_COMP': num_components,
            'K': k,
            'CLUSTER_TYPE': clustering_type,
            'SILHOUETTE_SCORE': silhouette_scores,
            'EXP_VAR': explained_variance
        })
        # cluster_df.to_csv(kmeans_file, index=False)

        # Find maximum
        ind = cluster_df.SILHOUETTE_SCORE.idxmax()
        best_clusters = cluster_df.loc[ind, :]

        # Run clustering
        # self.pca = PCA(n_components=best_clusters.loc['N_COMP'])
        # self.pca.fit(data_scaled)
        # data_pca = self.pca.transform(data_scaled)
        # exp_var = np.sum(pca.explained_variance_ratio_)
        if best_clusters.loc['CLUSTER_TYPE'] == 'KMeans':
            c = KMeans(n_clusters=best_clusters.loc['K'])
        elif best_clusters.loc['CLUSTER_TYPE'] == 'AgglomerativeClustering':
            c = AgglomerativeClustering(n_clusters=best_clusters.loc['K'])
        c.fit(data_scaled)
        pred = c.labels_
        curr_score = silhouette_score(data_scaled, pred)
        self.model = c
        self.silhouette_score = curr_score
        self.k = best_clusters.loc['K']
    
        print('Clustering done...')
        print(
            'Silhouette = ', self.silhouette_score, 'Method = ', best_clusters.loc['CLUSTER_TYPE'], 'N = ',
            best_clusters.loc['N_COMP'], 'EXP_VAR=', np.sum(exp_var), 'K = ', self.k
        )

        cluster = pd.DataFrame(pred, columns=['cluster'], index=data.index.values)
        cluster['study_id'] = data['study_id'].values
        # cluster.to_csv(kmeans_file.replace('scores_', ''), index=False)

        return self.silhouette_score, self.k, cluster, cluster_df

    def predict(self, data, features):
        """
        Predict the cluster

        :param data: np.array, the data
        """
        # return self.model.predict(self.pca(self.sc(data[features])))
        study_ids = data.study_id.values
        clusters = self.model.predict(self.sc.transform(data[features]))
        cluster_df = pd.DataFrame({'study_id': study_ids, 'cluster': clusters})
        cluster_df.index = data.index
        return cluster_df

    def get_num_clusters(self):
        """
        Return the number of clusters
        """
        return self.k


def calculate_sample_weights(train_data, val, features):
    """
    Create an auxillary classifier to get weighting
    on proxmity to the validation set.

    Idea is to correct for covariate shift.
    Assume already normalized.

    :param train_data: pd.DataFrame, the training data
    :param val: pd.DataFrame, the validation data
    :param features: list<str>, the features

    :return: np.array, the weights over the training data
    """
    # Add target col
    train_data['data'] = 1
    val['data'] = 0

    # Concatenate
    data = pd.concat([train_data, val]).reset_index(drop=True)

    # Make classifier
    rf = RandomForestClassifier()
    rf.fit(data[features], data['data'])

    # Now get prediction and probs
    probs = rf.predict_proba(train_data[features])

    # Calculate weights
    weights = (1 / probs[:, 1]) - 1
    weights /= np.mean(weights)
    return weights


def cohens_d(d1, d2):
    """
    Calculate cohens d
    Assume equal sample size

    (m2 - m1) / (np.sqrt((s1 + s2) / 2))

    :param d1: np.array, dataset 1
    :param d2: np.array, dataset 2
    """
    m1 = np.mean(d1)
    m2 = np.mean(d2)
    s1 = np.var(d1)
    s2 = np.var(d2)

    return (m1 - m2) / np.sqrt((s1 + s2) / 2)


def cohens_d_paired(df, col1, col2):
    """
    Calculate paired cohens d
    Assume equal sample size

    (meaan difference) / (standard dev difference)

    :param df: pd.DataFrame, dataframe
    :param col1: <str>, column
    :param col2: <str> column
    """
    m = np.mean(df[col1] - df[col2])
    std = np.std(df[col1] - df[col2])

    return m / std


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
