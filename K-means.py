import numpy as np
import random
import matplotlib as plt

from io_utilities import load_data
from visualizations import show_clusters_centroids

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

    data = load_data('iris.data')
    k = 3

    x = np.array([f[:-1] for f in data])
    y = np.array([f[-1] for f in data])

    clusters, centroids = k_means(x, k)

    show_clusters_centroids(clusters, centroids, "Result", keep = True)
    plt.show()
