import numpy as np

def _centroid(list_of_lists, n):
    centroids = []

    for sub_list in list_of_lists:
        centroid = None

        for el in sub_list[:n]:
            label, vector, weight = el
            if centroid is None:
                centroid = vector
            else:
                centroid += vector

        centroids.append(centroid)

    return np.sum(centroids, axis=0)

possible_functions = {'centroid': _centroid}