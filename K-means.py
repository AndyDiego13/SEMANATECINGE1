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

    mbk = MiniBatchKMeans(init='k-means++',n_clusters= len(x),batch_size=batch_size,n_init=10,max_no_improvement=10,verbose=0)
    t0 = time.time()
    mbk.fit(x)
    t_mini_batch = time.time()-t0

    #plot

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    k_means_cluster_centers = k_means.cluster_centers_
    order = pairwise_distances_argmin(k_means.cluster_centers_,mbk.cluster_centers_)
    mbk_means_cluster_centers = mbk.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(x,k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(x,mbk_means_cluster_centers)

    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(len(x)), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(x[my_members, 0], x[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))

    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(len(x)), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(x[my_members, 0], x[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %(t_mini_batch, mbk.inertia_))

    different = (mbk_means_labels == 4)
    ax = fig.add_subplot(1, 3, 3)

    for k in range(len(x)):
        different += ((k_means_labels == k) != (mbk_means_labels == k))

    identic = np.logical_not(different)
    ax.plot(x[identic, 0], x[identic, 1], 'w',
            markerfacecolor='#bbbbbb', marker='.')
    ax.plot(x[different, 0], x[different, 1], 'w',
            markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()
