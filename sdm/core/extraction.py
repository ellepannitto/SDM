"""
extraction.py: a python module for extracting lemmas frequencies and syntactic patterns from texts
(using multiprocessing technique).
"""

import functools
import logging
import uuid
import tqdm
import collections
import itertools
import multiprocessing as mp

from sdm.utils import os_utils as outils
from sdm.utils import data_utils as dutils
from sdm.utils import corpus_utils as cutils
from sdm.utils import pipeline_utils as putils
from sdm.utils import filesmerger as futils

logger = logging.getLogger(__name__)


def StreamPipeline(output_dir, input_data, list_of_workers=[2,2,2]):

    list_of_functions = [a, b, c]

    pipeline = putils.Pipeline(list_of_functions, list_of_workers)

    for result_list in dutils.grouper(pipeline.run(input_data)):
        # extract_patterns_stream(result_list)
        pass


class CoNLLPipeline:
    """
    A class used to parse CONLL texts and extract some statistics.
    """
    def __init__(self, output_dir, input_paths, acceptable_path, delimiter, batch_size=100, list_of_workers=[1,1,1]):
        """
        :param str output_dir: path/to/folder
        :param List[] input_paths:
        :param str: acceptable_path:
        :param str delimiter:
        :param batch_size
        :param list_of_workers
        """
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.delimiter = delimiter
        self.accepted_pos, self.accepted_rels = dutils.load_acceptable_labels_from_file(acceptable_path)

        self.list_of_functions = [outils.get_filenames,
                         functools.partial(cutils.CoNLLReader, self.delimiter),
                         functools.partial(cutils.DependencyBuilder, self.accepted_pos, self.accepted_rels)]
        self.pipeline = putils.Pipeline(self.list_of_functions, list_of_workers, batch_size)
        # outils.get_filenames: from directory to filenames
        # cutils.CoNLLReader: from filepath to list of sentences
        # cutils.DependencyBuilder: from sentence to representation head + deps

        
    def stats(self, batch_size_stats, w_thresh, workers):

        """
        :param int batch_size_stats:
        :param int w_thresh:
        :param int workers:
        """

        tmp_folder = outils.add_tmp_folder(self.output_dir)

        with mp.Pool(workers) as pool:
            iterator = dutils.grouper(self.pipeline.run(self.input_paths), batch_size_stats)
            pool_imap = pool.imap(functools.partial(extract_stats, tmp_folder), iterator)

            for _ in tqdm.tqdm(pool_imap, desc="STATS"):
                pass

        prefix_to_merge = ["lemma"]

        for prefix in prefix_to_merge:

            futils.merge(tmp_folder+"{}-freqs-*".format(prefix),
                         tmp_folder+"{}-merged".format(prefix),
                         mode=futils.Mode.txt)
            futils.collapse(tmp_folder+"{}-merged".format(prefix),
                            self.output_dir+"{}-freqs.txt".format(prefix), threshold=w_thresh)

        outils.remove(tmp_folder)

    def events(self, batch_size_events, e_thresh, workers):
        """
        :param int batch_size_events:
        :param int e_thresh:
        :param int workers:
        :return:
        :rtype:
        """

        # Load list of accepted words
        accepted_lemmas = dutils.load_lemmapos_freqs(self.output_dir+"lemma-freqs.txt")
        # accepted_lemmas = [tuple(i.split(" ")) for i in accepted_lemmas]
        # print(accepted_lemmas)
        # input()
        associative_relations = False

        tmp_folder = outils.add_tmp_folder(self.output_dir)

        with mp.Pool(workers) as pool:
            iterator = dutils.grouper(self.pipeline.run(self.input_paths), batch_size_events)
            pool_imap = pool.imap(functools.partial(extract_patterns, tmp_folder,
                                                    accepted_lemmas, associative_relations), iterator)

            for _ in tqdm.tqdm(pool_imap):
                pass

        prefix_to_merge = ["events", "n-events"]
        # for result_list in dutils.grouper(self.pipeline.run(self.input_paths), batch_size_events):
        # prefix_to_merge = extract_patterns(tmp_folder, result_list)
        # prefix_to_merge = extract_patterns(tmp_folder, result_list, accepted_lemmas=accepted_lemmas)

        for prefix in prefix_to_merge:
            futils.merge(tmp_folder+"{}-freqs-*".format(prefix),
                         tmp_folder+"{}-merged".format(prefix),
                         mode=futils.Mode.txt)
            futils.collapse(tmp_folder+"{}-merged".format(prefix),
                            self.output_dir+"{}-freqs.txt".format(prefix), threshold=e_thresh)

        outils.remove(tmp_folder)


def launchCoNLLPipeline(output_dir, input_paths, acceptable_path, delimiter, batch_size_stats, batch_size_events,
                        w_thres, e_thres, stats, events, list_of_workers):
    """

    :param str output_dir: path to output dir
    :param list input_paths: paths to input files/folders
    :param str acceptable_path: path to file containing acceptable pos in the 1st line and roles in the 2nd (space separated)
    :param str delimiter:
    :param int batch_size_stats:
    :param int batch_size_events:
    :param int w_thres:
    :param int e_thres:
    :param boolean stats:
    :param boolean events:
    :param list list_of_workers:
    """    
    
    conll_pip = CoNLLPipeline(output_dir, input_paths, acceptable_path,
                              delimiter, batch_size_stats, list_of_workers[:-1])

    if stats:
        conll_pip.stats(batch_size_stats, w_thres, list_of_workers[-1])
    if events:
        conll_pip.events(batch_size_events, e_thres, list_of_workers[-1])


def powerset(iterable):
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(2, len(iterable) + 1))

  
def extract_patterns(tmp_folder, accepted_lemmas, associative_relations, list_of_sentences):
    """

    :param str tmp_folder: path to temporary folder
    :param list_of_sentences: a tuple containing two objects: a words dictionary {token_id : {"lemma":lemma,"upos":pos}} and a dependencies dictionary {head_id:[(dep_id, role)]}
    :type list_of_sentences: (dict[str,dict], dict[str,list[tuple]])
    :param set accepted_lemmas: a list of accepted lemmas in the form {token_ID : {'lemma': lemma, 'upos': pos}..}
    :param boolean associative_relations:
    :return list: list

    """
    file_id = uuid.uuid4()
    events_freqdict = collections.defaultdict(int)

    if associative_relations:
        associative_events_freqdict = collections.defaultdict(int)

    groups = []
    for sentence, dependencies in filter(lambda x: x is not None, list_of_sentences):

        for head in dependencies:
            if head in sentence:
                group = set()
                if not len(accepted_lemmas) or (sentence[head]["lemma"], sentence[head]["upos"]) in accepted_lemmas:
                    group.add("{}@{}@{}".format(sentence[head]["lemma"], sentence[head]["upos"], "HEAD"))
                # print(dependencies[head])
                # input()

                for dep in dependencies[head]:
                    ide = dep[0]
                    synrel = dep[1]
                    if ide in sentence:
                        token = sentence[ide]
                        if not len(accepted_lemmas) or (token["lemma"], token["upos"]) in accepted_lemmas:
                            # print("ADDING TOKEN", token)
                            group.add("{}@{}@{}".format(token["lemma"], token["upos"], synrel))
                    else:
                        print("NOT FOUND IDE", ide)
                        # print("SENTENCE:", sentence)

                group = list(sorted(group))
                if len(group) < 16:
                    groups.append(group)
                # print("GROUP ADDED", groups)

            else:
                print("HEAD NOT IN SENTENCE", head)
                # print("SENTENCE:", sentence)

    for group in groups:
        # if len(group)>10:
        #     print(len(group), "-", group)
        subsets = powerset(group)
        for subset in subsets:
            events_freqdict[subset] += 1
            # print(events_freqdict)
            # input()

    if associative_relations:
        for group1, group2 in itertools.combinations(groups, r=2):
            cp = itertools.product([group1, group2])
            for el1, el2 in cp:
                associative_events_freqdict[(min(el1, el2), max(el1, el2))] += 1

    sorted_freqdict = sorted(events_freqdict.items(), key=lambda x: x[0])
    with open(tmp_folder + "events-freqs-{}".format(file_id), "w") as fout:
        for tup, freq in sorted_freqdict:
            # print(tup, freq)
            # input()
            print("{}\t{}".format(" ".join(tup), freq), file=fout)

    if associative_relations:
        sorted_freqdict = sorted(associative_events_freqdict.items(), key=lambda x: x[0])
        with open(tmp_folder + "associative-events-freqs-{}".format(file_id), "w") as fout:
            for tup, freq in sorted_freqdict:
                print("{}\t{}".format(" ".join(tup), freq), file=fout)

        # return ["events", "n-events", "associative-events"]

    # return ["events", "n-events"]

def extract_stats(tmp_folder, list_of_sentences):
    """

    :param str tmp_folder: path to temporary folder
    :param list_of_sentences: a tuple containing two objects: a words dictionary {token_id : {"lemma":lemma,"upos":pos}} and a dependencies dictionary {head_id:[(dep_id, role)]}
    :type list_of_sentences: (dict[str,dict], dict[str,list[tuple]])
    :return dictionary: dictionary of dictionaries {"lemma": { (lemma, pos): freq ..}}
    """

    file_id = uuid.uuid4()
    lemma_freqdict = collections.defaultdict(int)
    for sentence, _ in filter(lambda x: x is not None, list_of_sentences):
        for token_id in sentence:
            token = sentence[token_id]
            lemma, pos = token["lemma"], token["upos"]
            lemma_freqdict[(lemma,pos)] += 1

    dict_of_dicts = {"lemma": lemma_freqdict}

    for prefix, dic in dict_of_dicts.items():
        sorted_freqdict = sorted(dic.items(), key = lambda x: x[0])

        with open(tmp_folder+"{}-freqs-{}".format(prefix, file_id), "w") as fout:
            for tup, freq in sorted_freqdict:
                print("{}\t{}".format(" ".join(tup), freq), file=fout)

    # return dict_of_dicts.keys()