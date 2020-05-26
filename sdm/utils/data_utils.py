import numpy as np
import logging
import itertools


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

        if self.withPoS:
            form, pos = item
            # TODO: allow for different composition functions
            form = form+"/"+pos
        else:
            form = item
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


def load_set_freqs(filepath):
    ret = set()
    with open(filepath) as fin:
        for line in fin:
            line = line.strip().split("\t")
            ret.add(line[0])
    return ret


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

import heapq
import contextlib
import gzip
import glob

from enum import Enum


class Mode(Enum):
    txt = 1
    gzip = 2


def merge(filename_pattern, output_filename, mode=Mode.txt):

    openfunc = lambda fname: open(fname)
    if mode == Mode.gzip:
        openfunc = lambda fname: gzip.open(fname, "rt")

    files = glob.iglob(filename_pattern)

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(openfunc(fn)) for fn in files]
        with gzip.open(output_filename, 'wt') as f:
            f.writelines(heapq.merge(*files))


def collapse(filename, output_filename, delimiter="\t"):

    with gzip.open(filename, "rt") as fin, open(output_filename, "w") as fout:

        firstline, firstfreq = fin.readline().strip().split(delimiter)
        firstfreq = float(firstfreq)

        for line in fin:
            el, freq = line.strip().split(delimiter)
            freq = float(freq)
            if el == firstline:
                firstfreq += freq
            else:
                print("{}\t{}".format(firstline, firstfreq), file=fout)
                firstline = el
                firstfreq = freq

        print("{}\t{}".format(firstline, firstfreq), file=fout)