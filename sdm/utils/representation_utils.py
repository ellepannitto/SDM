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
        if centroid is not None:
            centroids.append(centroid)


    if len(centroids)>0:
        ret = np.sum(centroids, axis=0)
    else:
        ret = None
    return ret

possible_functions = {'centroid': _centroid}