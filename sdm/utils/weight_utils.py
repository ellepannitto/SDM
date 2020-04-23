from scipy.spatial import distance


def _cosine(list_of_items, head_vector):

    new_list_of_items = []
    for label, vector, w in list_of_items:
        score = 1-distance.cosine(vector, head_vector)
        new_list_of_items.append((label, vector, score))

    return new_list_of_items


possible_functions = {'cosine': _cosine}