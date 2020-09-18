import numpy as np
import random
import matplotlib.pyplot as plt
import time

from io_utilities import load_data
from visualizations import show_clusters_centroids
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


def dist(a, b):
    c = np.sum((np.array(b) - np.array(a))**2)
    return np.sqrt(c)


def cerc(puntos, centros):
    a = [[] for f in centros]

    for i, punto in enumerate(puntos):
        distnc = []

        for j, centro in enumerate(centros):
            #import pdb; pdb.set_trace()
            distnc.append(dist(punto, centro))
        argmin = np.argmin(distnc)

        a[argmin].append(punto)

    return a


def centros(a):
    b = []

    for a in a:
        centroid = np.mean(a, axis = 0)
        b.append(centroid)

    return b


def k_means(puntos, k, iter = 10):

    indx = np.random.randint(len(puntos), size = k)

    centroids = puntos[indx, :]
    clusters = cerc(puntos, centroids)

    for i in range(iter):
        clusters = cerc(puntos, centroids)
        centroids = centros(clusters)

    return clusters, centroids

if __name__ == "__main__":

    data = load_data("./data/iris.data")
    k = 3

    x = np.array([f[:-1] for f in data])
    y = np.array([f[-1] for f in data])
    
    clusters, centroids = k_means(x, k)

    show_clusters_centroids(clusters, centroids, "Result", keep = True)
    plt.show(show_clusters_centroids)

    k_means = KMeans(init='k-means++',n_clusters= len(x),n_init=10)
    t0 = time.time()
    k_means.fit(x)
    t_batch = time.time()-t0
    batch_size = 15

    #plot

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(left=0.05, right=2.0, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    k_means_cluster_centers = k_means.cluster_centers_

    k_means_labels = pairwise_distances_argmin(x,k_means_cluster_centers)

    ax = fig.add_subplot(1,3,1)
    for k, col in zip(range(len(x)), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(7.02, 3.0,  'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
    plt.show()
