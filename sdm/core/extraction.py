import functools
import tqdm
import logging
import uuid
import collections
import multiprocessing as mp

import os
import glob

from sdm.utils import os_utils as outils
from sdm.utils import data_utils as dutils
from sdm.utils import corpus_utils as cutils
from sdm.utils import pipeline_utils as putils

logger = logging.getLogger(__name__)


def extract_stats(list_of_sentences):
    pass

def extract_patterns_stream(list_of_sentences):
    pass

def extract_patterns(list_of_sentences):
    pass

def StreamPipeline(output_dir, input_data, list_of_workers=[2,2,2]):

    list_of_functions = [a, b, c]

    pipeline = putils.Pipeline(list_of_functions, list_of_workers)

    for result_list in dutils.grouper(pipeline.run(input_data)):
        extract_patterns_stream(result_list)


def CoNLLPipeline(output_dir, input_paths, list_of_workers = [2,2,2]):

    list_of_functions = [outils.get_filenames, cutils.CoNLLReader, cutils.DependencyBuilder]
    # outils.get_filenames: from directory to filenames
    # cutils.CoNLLReader: from filepath to list of sentences
    # cutils.DependencyBuilder: from sentence to representation head + deps

    pipeline = putils.Pipeline(list_of_functions, list_of_workers)

    for result_list in dutils.grouper(pipeline.run(input_paths)):
        extract_stats(result_list)

    for result_list in dutils.grouper(pipeline.run(input_paths)):
        extract_patterns(result_list)


if __name__ == "__main__":

    extract_stats("/home/ludovica.pannitto/prova_multiprocessing/",
                  ["/extra/corpora/corpora-en/ukWaC/3_UD/ukwac1/", "/extra/corpora/corpora-en/ukWaC/3_UD/ukwac2/"],
                  [outils.get_filenames, CoNLLReader, functools.partial(build_dependencies, params)])
