"""
extraction_w_pipeline.py: a python module for extracting lemmas frequencies and syntactic patterns from texts
(using multiprocessing technique).
"""

import functools
import logging
import uuid
import collections
import itertools
import shutil
import tempfile

import time

from sdm.utils import os_utils as outils
from sdm.utils import data_utils as dutils
from sdm.utils import corpus_utils as cutils
from sdm.utils import Pipeline as putils
from sdm.utils.FileMerger import filesmerger as fmutils

logger = logging.getLogger(__name__)


# def StreamPipeline(output_dir, input_data, list_of_workers=[2,2,2]):
#
#     list_of_functions = [a, b, c]
#
#     pipeline = putils.Pipeline(list_of_functions, list_of_workers)
#
#     for result_list in dutils.grouper(pipeline.run(input_data)):
#         # extract_patterns_stream(result_list)
#         pass


# class CoNLLPipeline:
#     """
#     A class used to parse CONLL texts and extract some statistics.
#     """
    # def __init__(self, output_dir, input_paths, acceptable_path, delimiter, accepted_lemmas,
    #              batch_size, list_of_workers):
    #     """
    #     :param str output_dir: path/to/folder
    #     :param List[] input_paths:
    #     :param str: acceptable_path:
    #     :param str: delimiter:
    #     :param batch_size
    #     :param list_of_workers
    #     """
    #     self.input_paths = input_paths
    #     self.output_dir = output_dir
    #     self.delimiter = delimiter
    #     self.accepted_pos, self.accepted_rels = dutils.load_acceptable_labels_from_file(acceptable_path)
    #
    #     tmp_path_merge = tempfile.mkdtemp(dir=self.output_dir)
    #     tmp_path_extraction = tempfile.mkdtemp(dir=self.output_dir)
    #
    #     state_class = fmutils.HierarchicalMerger(tmpdir=tmp_path_merge, delete_input=True)
    #
    #     self.list_of_functions = [outils.get_filenames,
    #                      functools.partial(cutils.CoNLLReader, self.delimiter),
    #                      functools.partial(cutils.DependencyBuilder, self.accepted_pos, self.accepted_rels),
    #                      functools.partial(extract_patterns, tmp_path_extraction, accepted_lemmas, False),
    #                      state_class.generator_add_for_pipeline]
    #
    #     self.pipeline = putils.Pipeline(self.list_of_functions, list_of_workers, batch_size)


def events_manager(output_dir, input_paths, acceptable_labels, delimiter, batch_size_list, e_thresh, w_thresh,
                   lemmas_freqs_file, workers, associative_relations):

    tmp_folder = tempfile.mkdtemp(dir=output_dir)+"/"
    # tmp_folder_events = tempfile.mkdtemp(dir=output_dir)+"/"
    # tmp_path_merge = tempfile.mkdtemp(dir=output_dir)+"/"

    accepted_pos, accepted_rels = dutils.load_acceptable_labels_from_file(acceptable_labels)

    accepted_lemmas = dutils.load_lemmapos_freqs(lemmas_freqs_file, w_thresh)

    state_class = fmutils.HierarchicalMerger(tmpdir=tmp_folder, delete_input=True)

    list_of_functions = [outils.get_filenames,
                         functools.partial(cutils.CoNLLReader, delimiter, batch_size_list[2]),
                         functools.partial(cutils.DependencyBuilder, accepted_pos, accepted_rels),
                         functools.partial(extract_patterns, tmp_folder, accepted_lemmas, associative_relations),
                         state_class.generator_add_for_pipeline]

    conll_pip = putils.Pipeline(list_of_functions, workers, batch_size_list)

    start_time = time.time()

    for last_file_list in conll_pip.run(input_paths):
        pass
    last_file_list = last_file_list[-1]

    end_time = time.time()

    logger.info("Finished pipeline.run() and extract_patterns: time elapsed {} seconds".format(end_time-start_time))

    prefix = "events"

    fmutils.merge_and_collapse_iterable(last_file_list, output_dir+"{}-freqs.txt".format(prefix),
                                        tmpdir=tmp_folder, delete_input=True, threshold=e_thresh)

    shutil.rmtree(tmp_folder)


def stats_manager(output_dir, input_paths, acceptable_labels, delimiter, batch_size_list, w_thresh, workers):

    tmp_folder = tempfile.mktemp(dir=output_dir)
    # tmp_folder_stats = tempfile.mktemp(dir=output_dir)
    # tmp_path_merge = tempfile.mktemp(dir=output_dir)

    accepted_pos, accepted_rels = dutils.load_acceptable_labels_from_file(acceptable_labels)

    state_class = fmutils.HierarchicalMerger(tmpdir=tmp_folder, delete_input=True)

    list_of_functions = [outils.get_filenames,
                         functools.partial(cutils.CoNLLReader, delimiter, batch_size_list[2]),
                         functools.partial(cutils.DependencyBuilder, accepted_pos, accepted_rels),
                         functools.partial(extract_stats, tmp_folder),
                         state_class.generator_add_for_pipeline]

    conll_pip = putils.Pipeline(list_of_functions, workers, batch_size_list)

    for _ in conll_pip.run(input_paths):
        pass

    logger.info("Finished pipeline.run() and extract_stats")

    prefix = "lemma"

    state_class.finalize(output_dir + "{}-freqs.txt".format(prefix), threshold=w_thresh)

    # TODO: check
    # futils.merge(tmp_folder + "{}-freqs-*".format(prefix),
    #              tmp_folder + "{}-merged".format(prefix),
    #              mode=futils.Mode.txt)
    # futils.collapse(tmp_folder + "{}-merged".format(prefix),
    #                 output_dir + "{}-freqs.txt".format(prefix), threshold=w_thresh)

    shutil.rmtree(tmp_folder)

    return output_dir + "{}-freqs.txt".format(prefix)


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
    yield [tmp_folder + "events-freqs-{}".format(file_id)]
    # yield [None]

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

    yield [tmp_folder+"lemma-freqs-{}".format(file_id)]
    # yield [None]

    # return dict_of_dicts.keys()