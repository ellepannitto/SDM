import functools
import logging
import uuid
import collections
import itertools


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


def CoNLLPipeline(output_dir, input_paths, delimiter, batch_size_stats, batch_size_events, list_of_workers = [1,1,1]):

    # accepted_pos, accepted_rels = dutils.load_parameter_from_file(file_passato_come_parametro)
    # print("CONLL PIPELINE", input_paths)
    # input()

    list_of_functions = [outils.get_filenames,
                         functools.partial(cutils.CoNLLReader, delimiter),
                         functools.partial(cutils.DependencyBuilder)]

    # outils.get_filenames: from directory to filenames
    # cutils.CoNLLReader: from filepath to list of sentences
    # cutils.DependencyBuilder: from sentence to representation head + deps

    pipeline = putils.Pipeline(list_of_functions, list_of_workers, batch_size_stats)

    tmp_folder = outils.add_tmp_folder(output_dir)
    for result_list in dutils.grouper(pipeline.run(input_paths), batch_size_stats):
        prefix_to_merge = extract_stats(tmp_folder, result_list)


    for prefix in prefix_to_merge:

        futils.merge(tmp_folder+"{}-freqs-*".format(prefix),
                     tmp_folder+"{}-merged".format(prefix),
                     mode=futils.Mode.txt)
        futils.collapse(tmp_folder+"{}-merged".format(prefix),
                        output_dir+"{}-freqs.txt".format(prefix), threshold=2)

    outils.remove(tmp_folder)

    # Load list of accepted words
    accepted_lemmas = dutils.load_set_freqs(output_dir+"lemma-freqs.txt")
    # print(accepted_lemmas)
    # input()

    tmp_folder = outils.add_tmp_folder(output_dir)
    for result_list in dutils.grouper(pipeline.run(input_paths), batch_size_events):
        # prefix_to_merge = extract_patterns(tmp_folder, result_list)
        prefix_to_merge = extract_patterns(tmp_folder, result_list, accepted_lemmas=accepted_lemmas)

    for prefix in prefix_to_merge:
        futils.merge(tmp_folder+"{}-freqs-*".format(prefix),
                     tmp_folder+"{}-merged".format(prefix),
                     mode=futils.Mode.txt)
        futils.collapse(tmp_folder+"{}-merged".format(prefix),
                        output_dir+"{}-freqs.txt".format(prefix), threshold=5)


def powerset(iterable):
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(2, len(iterable) + 1))

def extract_patterns(tmp_folder, list_of_sentences, accepted_lemmas=set(), associative_relations=False):

    file_id = uuid.uuid4()
    events_freqdict = collections.defaultdict(int)
    if associative_relations:
        associative_events_freqdict = collections.defaultdict(int)

    groups = []
    for sentence, dependencies in filter(lambda x: x is not None, list_of_sentences):

        for head in dependencies:
            if head in sentence:
                group = set()
                if not len(accepted_lemmas) or sentence[head]["lemma"] in accepted_lemmas:
                    group.add("{}@{}@{}".format(sentence[head]["lemma"], sentence[head]["upos"], "HEAD"))
                # print(dependencies[head])
                # input()

                for dep in dependencies[head]:
                    ide = dep[0]
                    synrel = dep[1]
                    if ide in sentence:
                        token = sentence[ide]
                        if not len(accepted_lemmas) or token["lemma"] in accepted_lemmas:
                            # print("ADDING TOKEN", token)
                            group.add("{}@{}@{}".format(token["lemma"], token["upos"], synrel))
                    else:
                        print("NOT FOUND IDE", ide)
                        # print("SENTENCE:", sentence)

                groups.append(list(sorted(group)))
                # print("GROUP ADDED", groups)

            else:
                print("HEAD NOT IN SENTENCE", head)
                # print("SENTENCE:", sentence)

    for group in groups:
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

        return ["events", "associative-events"]

    return ["events"]

def extract_stats(tmp_folder, list_of_sentences):

    file_id = uuid.uuid4()

    lemma_pos_rel_freqdict = collections.defaultdict(int)
    pos_rel_freqdict = collections.defaultdict(int)
    pos_freqdict = collections.defaultdict(int)
    rel_freqdict = collections.defaultdict(int)
    lemma_freqdict = collections.defaultdict(int)

    for sentence, _ in filter(lambda x: x is not None, list_of_sentences):
        for token_id in sentence:
            token = sentence[token_id]
            lemma, pos, rel = token["lemma"], token["upos"], token["deprel"]

            lemma_pos_rel_freqdict[(lemma, pos, rel)] += 1
            pos_rel_freqdict[(pos, rel)] += 1
            pos_freqdict[(pos,)] += 1
            rel_freqdict[(rel,)] += 1
            lemma_freqdict[(lemma,)] += 1


    dict_of_dicts = {"lemma-pos-rel":lemma_pos_rel_freqdict,
                     "pos-rel": pos_rel_freqdict,
                     "pos": pos_freqdict,
                     "rel": rel_freqdict,
                     "lemma": lemma_freqdict}

    for prefix, dic in dict_of_dicts.items():
        sorted_freqdict = sorted(dic.items(), key = lambda x: x[0])

        with open(tmp_folder+"{}-freqs-{}".format(prefix, file_id), "w") as fout:
            for tup, freq in sorted_freqdict:
                print("{}\t{}".format(" ".join(tup), freq), file=fout)

    return dict_of_dicts.keys()