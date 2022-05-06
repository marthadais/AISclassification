import numpy as np
from sklearn.preprocessing import LabelEncoder
from math import dist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score

#https://douglasrizzo.com.br/dunn-index/


def all_distances(labels, features, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)
    d_all = euclidean_distances(features)

    for i in np.arange(0, len(labels) - 1):
        ii_list = np.where(labels != labels[i])[0]
        for ii in ii_list:
            d_iii = d_all[i, ii]
            if d_iii < cluster_distances[labels[i], labels[ii]]:
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = d_iii
        ii_list = np.where(labels == labels[i])[0]
        for ii in ii_list:
            if d_all[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = d_iii
            # diameters[labels[i]] += d_all[i, ii]

    # for i in range(len(diameters)):
    #     diameters[i] /= sum(labels == i)

    return cluster_distances, diameters


def dunn_index(labels, features):
    labels = LabelEncoder().fit(labels).transform(labels)
    ic_distances, ic_diameter = all_distances(labels, features)
    if ic_distances.sum() == 0:
        min_distance = 0
    else:
        min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(ic_diameter)
    return min_distance / max_diameter


def bootstrap_sampling(features, labels, times=50):
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    aux = features.index
    k = int(len(labels) * 0.005)
    dunn = []
    sil = []
    for i in range(times):
        idx_curr = np.random.choice(aux, replace=False, size=k)
        dunn.append(dunn_index(labels.loc[idx_curr], features.loc[idx_curr, :]))
        sil.append(silhouette_score(features.loc[idx_curr, :], labels.loc[idx_curr]))
    dunn_avg = np.array(dunn).mean()
    sil_avg = np.array(sil).mean()
    print(f'DUNN index data = {dunn_avg}')
    print(f'Silhouette index data = {sil_avg}')

    return dunn_avg, sil_avg


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    data = load_iris()
    kmeans = KMeans(n_clusters=3)
    c = data['target']
    x = data['data']
    k = kmeans.fit_predict(x)
    d = euclidean_distances(x)

    dund = dunn_index(c, x)
    dunk = dunn_index(k, x)
    print(dund, dunk)
