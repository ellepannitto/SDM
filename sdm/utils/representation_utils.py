import numpy as np

def _centroid(list_of_lists, n):
    centroids = []

    for sub_list in list_of_lists:
        centroid = None
        n_summed = 0

        for el in sub_list[:n]:
            label, vector, weight = el
            if centroid is None:
                centroid = vector
                n_summed += 1
            else:
                centroid += vector
                n_summed += 1

        if centroid is not None:
            centroid = centroid / n_summed
            centroids.append(centroid)


    if len(centroids)>0:
        ret = np.sum(centroids, axis=0)
        ret = ret / len(centroids)
    else:
        ret = None
    return ret

possible_functions = {'centroid': _centroid}