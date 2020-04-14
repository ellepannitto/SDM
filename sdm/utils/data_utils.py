import numpy as np

def load_dataset(fpath):
    dataset = []
    with open(fpath) as fin:
        for line in fpath:
            ac_content = set()
            item, object_relation = line.split("\t")
            item = item.split()
            elements = []
            for el in item:
                form, pos, rel = el.split("@")
                ac_content.add(rel)
                elements.append((form, pos, rel))
            dataset.append([elements, object_relation, ac_content])

    return dataset


def load_mapping(fpath):
    ret = {}
    with open(fpath) as fin:
        for line in fpath:
            line = line.strip.split("\t")
            ret[line[0]] = line[1].split()

    return ret


def load_vectors(model_fpath, noun_set=set()):
    noun_vectors = {}

    with open(model_fpath) as fin_model:
        fin_model.readline()
        for line in fin_model:
            line = line.strip().split()
            word = line[0]
            if word in noun_set or not len(noun_set):
                try:
                    vector = [float(x) for x in line[1:]]
                    noun_vectors[word] = vector
                except:
                    print("problem with vector for word", word)

    min_len = min([len(vector) for vector in noun_vectors.values()])
    print("LEN VECTORS: ", min_len)
    for n in noun_vectors:
        v = noun_vectors[n][-min_len:]
        noun_vectors[n] = np.array(v)

    return noun_vectors


def load_set(filepath):
    ret = set()
    with open(filepath) as fin:
        for line in fin:
            line = line.strip()
            ret.add(line)
    return ret