"""
Entry point of Structured Distributional Model
"""

import os
import logging.config
import argparse

import sdm.utils.config as cutils
import sdm.utils.graph_utils as gutils
import sdm.utils.os_utils as outils

import sdm.core.model as model

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig( config_dict )

logger = logging.getLogger(__name__)

def _build_representations(args):
    output_path = outils.check_dir(args.output_dir)
    uri = args.uri
    username = args.user
    password = args.password
    relations_fpath = args.relations
    data_fpaths = args.data
    vector_fpath = args.vectors

    graph = gutils.connect_to_graph(uri, username, password)

    model.build_representation(output_path, graph, relations_fpath, data_fpaths, vector_fpath)


def _extract_relations_list(args):
    output_path = outils.check_dir(args.output_dir)
    uri = args.uri
    username = args.user
    password = args.password

    graph = gutils.connect_to_graph(uri, username, password)

    gutils.extract_relations_list(output_path, graph)


def main():
    """Launch SDM"""

    parser = argparse.ArgumentParser(prog='sdm')
    subparsers = parser.add_subparsers()

    parser_relationlist = subparsers.add_parser('extract-possible-relations',
                                              help='extract possible relations from graph')
    parser_relationlist.add_argument("-o", "--output-dir",
                                     help="path to output dir, default is data/results/")
    parser_relationlist.add_argument("-U", "--uri", help="uri to connect to the graph")
    parser_relationlist.add_argument("-u", "--user", help="user to connect to the graph")
    parser_relationlist.add_argument("-p", "--password", help="password to connect to the graph")
    parser_relationlist.set_defaults(func=_extract_relations_list)

    parser_build = subparsers.add_parser('build-representations',
                                                        help='build representations over input files')
    parser_build.add_argument("-o", "--output-dir",
                                     help="path to output dir, default is data/results/")
    parser_build.add_argument("-U", "--uri", help="uri to connect to the graph")
    parser_build.add_argument("-u", "--user", help="user to connect to the graph")
    parser_build.add_argument("-p", "--password", help="password to connect to the graph")
    parser_build.add_argument("-r", "--relations", required=True,
                              help="path to file containing mapping for relations")
    parser_build.add_argument("-d", "--data", nargs='+', required=True,
                              help='paths to files containing dataset')
    parser_build.add_argument("-v", "--vectors", required=True,
                              help='path to file containing vectors')
    parser_build.set_defaults(func=_build_representations)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()