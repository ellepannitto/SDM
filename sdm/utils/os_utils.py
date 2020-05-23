import logging
import os
import shutil
import glob

logger = logging.getLogger(__name__)


def add_tmp_folder(path, exist_ok=True):
    tmp_path = path+"/tmp/"
    os.makedirs(tmp_path, exist_ok=exist_ok)
    return tmp_path


def remove(path):
    shutil.rmtree(path)


def check_dir(path):
    logger.info("Writing output to: {}".format(path))
    path = path+"/"
    os.makedirs(path, exist_ok=True)
    return path


def get_filenames(input_path):

    if os.path.isfile(input_path):
        yield input_path
    else:
        for filename in glob.glob(input_path+"/*"):
            if os.path.isfile(filename):
                yield filename