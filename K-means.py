import numpy as np

def dist(a, b):
    c = 0

    for i in range(len(a)):
        c += (b[i] - a[i])**2
    return np.sqrt(c)

def cerc(puntos, centros):
    a = [[] for f in centros]

    for i, punto in enumerate(puntos):
        this_distance = []
        for j,centro in enumerate(centros):
            this_distance.append(dist((punto, centro)))
        this_argmin = np.argmin(this_distance)
        a[this_argmin].append(punto)

    return a


def centros(a):
    b = []
    c= 0
    x = np.array()

    for i in range(len(a)):
        for j in range(len(a[i])):
            x.append(np.array(a[i][j]))    #[[x, y, z], [x, y, z]] x = [x1, x2, xn]
        c = c/len(a[i])
        b.append(c)
    return b


def k_means(puntos):
    k = np.random.rand()

    puntos.append(k)