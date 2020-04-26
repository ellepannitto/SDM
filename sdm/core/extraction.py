import functools
import tqdm
import logging
import uuid
from multiprocessing import Pool

from sdm.utils import os_utils as outils
from sdm.utils import data_utils as dutils
from sdm.utils import corpus_utils as cutils

logger = logging.getLogger(__name__)


def sentences_generator(files_list):
    for filename in files_list:
        for sentence in cutils.ukWaCReader(filename):
            yield sentence


def parallel_process_sentences(output_path, list_of_sentences):
    tmp_id = uuid.uuid4()

    # TODO: move to file
    accepted_ne = ["PERSON", "ORGANIZATION", "LOCATION", "COUNTRY", "CITY", "STATE_OR_PROVINCE"]
    accepted_pos = ["N", "V", "J"]
    accepted_synrels = []

    for sentence in list_of_sentences:
        for token in sentence:
            tokensplit = token.split()
            try:
                id, form, lemma, pos, ne, synrels = tokensplit
                synrels = synrels.split(",")
                split_synrels = []
                for rel in synrels:
                    label, head = rel.split("=")
                    split_synrels.append((label, head))
                # if pos in ["NNP", "NNPS"] and ne in accepted_ne:

            except:
                print(token)


def extract_stats(output_dir, input_paths, n_workers, batch_size):
    list_of_files = outils.get_filenames(input_paths)

    tmp_path = outils.add_tmp_folder(output_dir)

    with Pool(n_workers) as p:
        iterator = dutils.grouper(sentences_generator(list_of_files), batch_size)
        imap_obj = p.imap(functools.partial(parallel_process_sentences, tmp_path), iterator)

        for _ in tqdm.tqdm(imap_obj, total=len(list_of_files) // batch_size):
            pass


    outils.remove(tmp_path)