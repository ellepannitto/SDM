import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_dataset(fpath):
    dataset = []
    with open(fpath) as fin:
        header = fin.readline().strip().split()

        item_col = header.index("item")
        target_col = header.index("target_relation")

        for line in fin:
            ac_content = set()
            linesplit = line.strip().split("\t")
            if len(linesplit):
                item, object_relation = linesplit[item_col], linesplit[target_col]
                ac_content.add(object_relation)
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
        for line in fin:
            line = line.strip().split()
            ret[line[0]] = line[1].split()

    return ret


def load_vectors(model_fpath, noun_set=set(), len_vectors = -1):
    noun_vectors = {}

    with open(model_fpath) as fin_model:
        n_words, len_from_file = fin_model.readline().strip().split()
        len_from_file = int(len_from_file)
        if len_vectors == -1:
            len_vectors = len_from_file
        for line in fin_model:
            line = line.strip().split()
            len_line = len(line)
            word = " ".join(line[:len_line-len_from_file])
            if word in noun_set or not len(noun_set):
                try:
                    vector = [float(x) for x in line[-len_vectors:]]
                    noun_vectors[word] = np.array(vector)
                    # print(word)
                    # print(noun_vectors[word])
                    # input()
                except:
                    logger.info("problem with vector for word {}".format(word))

    # min_len = min([len(vector) for vector in noun_vectors.values()])
    # logger.info("LEN VECTORS: {}".format(min_len))
    # for n in noun_vectors:
    #     v = noun_vectors[n][-min_len:]
    #     noun_vectors[n] = np.array(v)

    logger.info("loaded {} vectors".format(len(noun_vectors)))
    return noun_vectors


def load_set(filepath):
    ret = set()
    with open(filepath) as fin:
        for line in fin:
            line = line.strip()
            ret.add(line)
    return ret