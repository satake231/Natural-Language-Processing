import itertools
import logging
import pickle
import random

import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

from collections import defaultdict
from deprecated import deprecated
from scipy.spatial.distance import cdist
from tqdm import tqdm
from string import ascii_uppercase
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizer, BertJapaneseTokenizer

# random_stateで使う定数
SEED = 42
#%%
def best_kmeans(X, max_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best K-Means clustering given the data, a range of K values, and a K-selection criterion.

    :param X: usage matrix (made of usage vectors)
    :param max_range: range within the number of clusters should lie
    :param criterion: K-selection criterion: 'silhouette' or 'calinski'
    :return: best_model: KMeans model (sklearn.cluster.Kmeans) with best clustering according to the criterion
             scores: list of tuples (k, s) indicating the clustering score s obtained using k clusters
    """
    assert criterion in ['silhouette', 'calinski', 'harabasz', 'calinski-harabasz']
    best_model, best_score = None, -1
    scores = []
    # クラスター数2から11までの間で最もシルエットスコアが高いものを選択
    for k in max_range:
        print(k)
        if k < X.shape[0]:
            # クラスター数に応じてKmeansを行う
            kmeans = KMeans(n_clusters=k, random_state=SEED)
            cluster_labels = kmeans.fit_predict(X)
            # シルエットスコアを算出
            if criterion == 'silhouette':
                score = silhouette_score(X, cluster_labels)
            else:
                score = calinski_harabasz_score(X, cluster_labels)

            scores.append((k, score))

            # if two clusterings yield the same score, keep the one that results from a smaller K
            if score > best_score:
                best_model, best_score = kmeans, score
    return best_model, scores
#%%
def cluster_usages(Uw, method='kmeans', k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best clustering model for a usage matrix.

    :param Uw: usage matrix
    :param method: K-Means or Gaussian Mixture Model ('kmeans' or 'gmm')
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: best clustering model
    """
    # standardize usage matrix by removing the mean and scaling to unit variance
    X = preprocessing.StandardScaler().fit_transform(Uw)

    # get best model according to a K-selection criterion
    if method == 'kmeans':
        best_model, _ = best_kmeans(X, k_range, criterion=criterion)
    elif method == 'gmm':
        best_model_aic, best_model_bic, _, _ = best_gmm(X, k_range)
        if criterion == 'aic':
            best_model = best_model_aic
        elif criterion == 'bic':
            best_model = best_model_bic
        else:
            raise ValueError('Invalid criterion {}. Choose "aic" or "bic".'.format(criterion))
    else:
        raise ValueError('Invalid method "{}". Choose "kmeans" or "gmm".'.format(method))

    return best_model
#%%
def obtain_clusterings(usages, out_path, method='kmeans', k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return and save dictionary mapping lemmas to their best clustering model, given a method-criterion pair.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param out_path: output path to store clustering models
    :param method: K-Means or Gaussian Mixture Model ('kmeans' or 'gmm')
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: dictionary mapping lemmas to their best clustering model
    """
    clusterings = {}  # dictionary mapping lemmas to their best clustering
    for w in tqdm(usages):
        print(w)
        Uw = []
        # print(len(usages[w]))
        # 文章数*768次元データの作成
        for i in range(len(usages[w])):
            Uw_l, _, _, _ = usages[w][i]
            Uw.append(Uw_l)
        Uw = np.stack(Uw, axis=0)
        # データ構造を確認
        print(Uw.shape)
        clusterings[w] = cluster_usages(Uw, method, k_range, criterion)

    with open(out_path, 'wb') as f:
        pickle.dump(clusterings, file=f)

    return clusterings
#%%
def to_one_hot(y, num_classes=None):
    """
    Transform a list of categorical labels into the list of corresponding one-hot vectors.
    E.g. [2, 3, 1] -> [[0,0,1,0], [0,0,0,1], [0,1,0,0]]

    :param y: N-dimensional array of categorical class labels
    :param num_classes: the number C of distinct labels. Inferred from `y` if unspecified.
    :return: N-by-C one-hot label matrix
    """
    if num_classes:
        K = num_classes
    else:
        K = np.max(y) + 1

    one_hot = np.zeros((len(y), K))

    for i in range(len(y)):
        one_hot[i, y[i]] = 1

    return one_hot


def usage_distribution(predictions, time_labels):
    """

    :param predictions: The clustering predictions
    :param time_labels:
    :return:
    """
    if predictions.ndim > 2:
        raise ValueError('Array of cluster probabilities has too many dimensions: {}'.format(predictions.ndim))
    if predictions.ndim == 1:
        predictions = to_one_hot(predictions)

    label_set = sorted(list(set(time_labels)))
    t2i = {t: i for i, t in enumerate(label_set)}

    n_clusters = predictions.shape[1]
    n_periods = len(label_set)
    usage_distr = np.zeros((n_clusters, n_periods))

    for pred, t in zip(predictions, time_labels):
        usage_distr[:, t2i[t]] += pred.T

    return usage_distr
#%%
with open('usages_16_len256_2014_12.dict', 'rb') as f:
    usages = pickle.load(f)
#%%
clusterings = obtain_clusterings(
    usages,
    out_path='usages_len128.clustering.dict',
    method='kmeans',
    k_range = np.arange(2, 11),
    criterion='silhouette'
)
print(clusterings)
#%%
with open("clusterings_kusa_4.dict", "wb") as tf:
    pickle.dump(clusterings,tf)
#%%
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
word = '草'
t_labels = []
for line in usages[word]:
    t_labels.append(line[3])
best_model = clusterings[word]

# create usage distribution based on clustering results
# クラスタリング結果を出力
print(best_model.labels_)
usage_distr = usage_distribution(best_model.labels_, t_labels)

cl_num = max(best_model.labels_) + 1
cluster_labels = best_model.labels_
print('クラスタ数: {}'.format(cl_num))

cluster_to_sentences = defaultdict(list)

# 各クラスターから例文を取得
for cluster_label, usage in zip(cluster_labels, usages[word]):
    # 文章idから元文にデコード
    sentence = ''.join(tokenizer.decode(usage[1], skip_special_tokens=True).split(' '))
    cluster_to_sentences[cluster_label].append(sentence)

# 各クラスターから例文を10個ずつ出力
for sentencs in cluster_to_sentences.values():
    print(sentencs[:10])
print(usage_distr)
#%%
