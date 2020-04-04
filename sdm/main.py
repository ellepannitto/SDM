"""
Entry point of Structured Distributional Model
"""

import os
import logging
import logging.config
import argparse

import sdm.utils.config as cutils

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig( config_dict )

logger = logging.getLogger(__name__)


def main():
    """Launch SDM"""

    parser = argparse.ArgumentParser(prog='sdm')
    subparsers = parser.add_subparsers()

    parser_uno = subparsers.add_parser()

    args = parser.parse_args()
    args.func(args)