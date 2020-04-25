import numpy as np
import logging

logger = logging.getLogger(__name__)


def dump_results(fpath, res, out_fpath):
    with open(fpath) as fin, open(out_fpath, "w") as fout:
        header = fin.readline().strip()
        print(header+"\tLC_vector\tAC_vector", file=fout)
        for i, line in enumerate(fin):
            line = line.strip()
            if res[i][0] is None:
                v1 = "None"
            else:
                v1 = " ".join("{:.3f}".format(x) for x in res[i][0])

            if res[i][1] is None:
                v2 = "None"
            else:
                v2 = " ".join("{:.3f}".format(x) for x in res[i][1])
            print(line+"\t{}\t{}".format(v1, v2), file=fout)


def load_dataset(fpath):
    dataset = []
    with open(fpath) as fin:
        header = fin.readline().strip().split()

        item_col = header.index("item")
        target_col = header.index("target-relation")

        for line in fin:
            ac_content = set()
            linesplit = line.strip().split("\t")
            if len(linesplit):
                item, object_relation = linesplit[item_col], linesplit[target_col]
                if not object_relation == 'SENTENCE':
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
            ret[line[0]] = line[1].split(",")

    return ret


class VectorsDict(dict):

    def __init__(self, withPoS=False):
        self.withPoS = withPoS

    def __getitem__(self, item):
        form, pos = item
        if self.withPoS:
            # TODO: allow for different composition functions
            form = form+"/"+pos
        return super().__getitem__(form)


def _load_vocab(fpath):
    ret = []
    with open(fpath) as fin:
        for line in fin:
            line = line.strip().split()
            for el in line:
                ret.append(el)

    return ret


def _load_vectors_npy(vectors_fpath, withPoS, noun_set, len_vectors):

    vectors_vocab = vectors_fpath[:-4]+".vocab"

    vectors = np.load(vectors_fpath)
    vocab = _load_vocab(vectors_vocab)

    noun_vectors = VectorsDict(withPoS)

    for key, value in zip(vocab, vectors):

        if key in noun_set or not len(noun_set):
            noun_vectors[key] = value
            if len_vectors > -1:
                noun_vectors[key] = value[:len_vectors]

    logger.info("loaded {} vectors".format(len(noun_vectors)))
    return noun_vectors


def _load_vectors_from_text(vectors_fpath, withPoS, noun_set, len_vectors):

    noun_vectors = VectorsDict(withPoS)

    with open(vectors_fpath) as fin_model:
        n_words, len_from_file = fin_model.readline().strip().split()
        len_from_file = int(len_from_file)

        for line in fin_model:
            if len_vectors == -1:
                len_vectors = len_from_file

            line = line.strip().split()
            len_line = len(line)
            word = " ".join(line[:len_line-len_from_file])

            if word in noun_set or not len(noun_set):
                try:
                    vector = [float(x) for x in line[-len_vectors:]]
                    noun_vectors[word] = np.array(vector)
                except:
                    logger.info("problem with vector for word {}".format(word))

    logger.info("loaded {} vectors".format(len(noun_vectors)))
    return noun_vectors

def load_vectors(vectors_fpath, withPoS=False, noun_set=set(), len_vectors=-1):

    if vectors_fpath.endswith(".npy"):
        ret = _load_vectors_npy(vectors_fpath, withPoS=withPoS, noun_set=noun_set, len_vectors=len_vectors)
    else:
        ret = _load_vectors_from_text(vectors_fpath, withPoS=withPoS, noun_set=noun_set, len_vectors=len_vectors)
    return ret


def load_set(filepath):
    ret = set()
    with open(filepath) as fin:
        for line in fin:
            line = line.strip()
            ret.add(line)
    return ret
