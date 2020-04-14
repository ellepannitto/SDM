import logging
import os
import glob

logger = logging.getLogger(__name__)


def check_dir(path):
    logger.info("Writing output to: {}".format(path))
    path = path+"/"
    os.makedirs(path, exist_ok=True)
    return path