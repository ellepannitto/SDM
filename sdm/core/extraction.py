import functools
import logging

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


def CoNLLPipeline(output_dir, input_paths, delimiter="\t", batch_size = 10000, list_of_workers = [2,2,2]):

    accepted_pos, accepted_rels = dutils.load_parameter_from_file(file_passato_come_parametro)

    list_of_functions = [outils.get_filenames,
                         functools.partial(cutils.CoNLLReader, delimiter),
                         functools.partial(cutils.DependencyBuilder, accepted_pos, accepted_rels)]

    # outils.get_filenames: from directory to filenames
    # cutils.CoNLLReader: from filepath to list of sentences
    # cutils.DependencyBuilder: from sentence to representation head + deps

    pipeline = putils.Pipeline(list_of_functions, list_of_workers)

    for result_list in dutils.grouper(pipeline.run(input_paths), batch_size):
        extract_stats(result_list)

    # Load list of accepted words

    for result_list in dutils.grouper(pipeline.run(input_paths), batch_size):
        extract_patterns(result_list)
