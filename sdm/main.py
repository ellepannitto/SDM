"""
Entry point of Structured Distributional Model
"""

import os
import logging.config
import argparse

import sdm.utils.config as cutils
import sdm.utils.graph_utils as gutils
import sdm.utils.os_utils as outils
import sdm.utils.data_utils as dutils

import sdm.core.model as model

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _build_representations(args):
    output_path = outils.check_dir(args.output_dir)
    uri = args.uri
    username = args.user
    password = args.password
    relations_fpath = args.relations
    data_fpaths = args.data
    vector_fpath = args.vectors

    weight_function = args.weight_function
    rank_forward = args.not_rank_forward
    rank_backward = args.not_rank_backward
    _N = args.N_from_graph
    _M = args.M_build_rep
    representation_function = args.representation_function
    include_same_relations = args.include_same_relations
    reduced_vec_len = args.reduced_vec_len
    vectors_with_PoS = args.vectors_with_PoS

    weight_to_extract = args.weight_from_graph

    graph = gutils.connect_to_graph(uri, username, password)

    vector_space = dutils.load_vectors(vector_fpath, withPoS=vectors_with_PoS,
                                       len_vectors=reduced_vec_len)

    model.build_representation(output_path=output_path, graph=graph, relations_fpath=relations_fpath,
                               data_fpaths=data_fpaths, vector_space=vector_space,
                               weight_function=weight_function, rank_forward=rank_forward, rank_backward=rank_backward,
                               N=_N, M=_M, include_same_relations=include_same_relations,
                               representation_function=representation_function,
                               weight_to_extract=weight_to_extract)


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
    parser_build.add_argument("--weight-function", default="cosine", choices=['cosine'],
                              help='function used to rank weighted list against representation vector')
    parser_build.add_argument("--not-rank-forward", action='store_false')
    parser_build.add_argument("--not-rank-backward", action='store_false')
    parser_build.add_argument("-N", "--N-from-graph", default=50, type=int,
                              help='number of elements to be retrieved from graph for each query')
    parser_build.add_argument("-M", "--M-build-rep", default=20, type=int,
                              help='number of elements to be considered to build representation '
                                   '(head of the weighted lists)')
    parser_build.add_argument("--representation-function", default="centroid", choices=['centroid'],
                              help='function used to build representation vector')
    parser_build.add_argument("--include-same-relations", action="store_true")
    parser_build.add_argument("--weight-from-graph", default='pmi', choices=['pmi', 'lmi'])
    parser_build.add_argument("--reduced-vec-len", type=int, default=-1)
    parser_build.add_argument("--vectors_with_PoS", action="store_true")

    parser_build.set_defaults(func=_build_representations)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
