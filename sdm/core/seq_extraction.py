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

def events_manager(output_dir, input_paths, acceptable_labels, delimiter, batch_size_farm, batch_size_merge,
                   e_thresh, w_thresh, lemmas_freqs_file, workers, associative_relations):

    tmp_folder = tempfile.mkdtemp(dir=output_dir)+"/"
    # tmp_folder_events = tempfile.mkdtemp(dir=output_dir)+"/"
    # tmp_path_merge = tempfile.mkdtemp(dir=output_dir)+"/"

    accepted_pos, accepted_rels = dutils.load_acceptable_labels_from_file(acceptable_labels)

    accepted_lemmas = dutils.load_lemmapos_freqs(lemmas_freqs_file, w_thresh)


    list_of_functions = [functools.partial(cutils.CoNLLReader, delimiter),
                         functools.partial(cutils.DependencyBuilder, accepted_pos, accepted_rels),
                         functools.partial(extract_patterns, tmp_folder, accepted_lemmas, associative_relations)]

    # MULTIPROCESSING, not hierarchical
    conll_pip = putils.Farm(list_of_functions, workers, batch_size_farm)
    reduce_fn = functools.partial(fmutils.merge_and_collapse_iterable, output_filename=None, tmpdir=tmp_folder,
                                   delete_input=True)

    # NOT MULTIPROCESSING, HIERARCHICAL
    # conll_pip = putils.Farm(list_of_functions, 1, batch_size_farm)
    # state_class = fmutils.HierarchicalMerger(batch=batch_size_merge, tmpdir=tmp_folder, delete_input=True)
    # reduce_fn = state_class.add

    start_time = time.time()

    merged_fname = conll_pip.map_reduce(outils.get_filenames(input_paths), reduce_fn, batch_size_merge)

    end_time = time.time()
    logger.info("Finished pipeline.run() and extract_patterns: time elapsed {} seconds".format(end_time-start_time))

    output_fname = output_dir+"/{}-freqs.txt".format("events")

    # MULTIPROCESSING, not hierarchical
    shutil.move (merged_fname, output_fname)

    # NOT MULTIPROCESSING, HIERARCHICAL
    # state_class.finalize(output_fname, threshold=e_thresh)


    # shutil.rmtree(tmp_folder)


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