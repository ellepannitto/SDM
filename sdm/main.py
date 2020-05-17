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

import sdm.core.datasets as datasets
import sdm.core.model as model
import sdm.core.extraction as extraction

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)


def _pipeline_extraction(args):
    output_path = outils.check_dir(args.output_dir)
    pipeline = args.pipeline

    if pipeline == "conll":
        extraction.CoNLLPipeline(output_path)
    elif pipeline == "stream":
        extraction.StreamPipeline(output_path)


def _compute_evaluation(args):
    output_path = outils.check_dir(args.output_dir)
    original_data = args.data
    generated_data = args.generated_data
    vecs = args.vectors
    mapfile = args.map
    data_type = args.type
    evaluation.compute_evaluation(original_data, generated_data, output_path, vecs, mapfile, data_type)


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


def _prepare_input(args):
    data = args.data
    outfolder = outils.check_dir(args.output_dir)
    datatype = args.type
    outtype = args.sequence_order
    datasets.prepare_input_files(data, outfolder, datatype, outtype)


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

    # Building the Graph

    parser_pipelineExtraction = subparsers.add_parser("pipeline-extraction",
                                                      help='from text to events')
    parser_pipelineExtraction.add_argument("-p", "--pipeline",
                                           help="type of pipeline to run",
                                           choices=["conll", "stream"], default="conll")
    parser_pipelineExtraction.add_argument("-o", "--output-dir", required=True)
    parser_pipelineExtraction.set_defaults(func=_pipeline_extraction)

    # TODO: from pipeline output to neo4j input format
    # TODO: import in neo4j
    # TODO: add weights
    # TODO: vectors?

    # SDM Model

    parser_relationlist = subparsers.add_parser('extract-possible-relations',
                                                help='extract possible relations from graph')
    parser_relationlist.add_argument("-o", "--output-dir",
                                     help="path to output dir, default is data/results/")
    parser_relationlist.add_argument("-U", "--uri", help="uri to connect to the graph")
    parser_relationlist.add_argument("-u", "--user", help="user to connect to the graph")
    parser_relationlist.add_argument("-p", "--password", help="password to connect to the graph")
    parser_relationlist.set_defaults(func=_extract_relations_list)

    parser_prepareInputFile = subparsers.add_parser('prepare-input',
                                                    help='take original dataset and convert in SDM accepted format')
    parser_prepareInputFile.add_argument("-d", "--data", nargs='+', required=True,
                                         help='paths to files containing dataset')
    parser_prepareInputFile.add_argument("-o", "--output-dir",
                              help="path to output dir, default is data/results/")
    parser_prepareInputFile.add_argument("-t", "--type", required=True, choices=["ks", "dtfit", "tfit_mit"], help="dataset type")
    parser_prepareInputFile.add_argument("--sequence-order", choices=["verbs_args", "head_verbs_args"],
                                         help="output arguments order")
    parser_prepareInputFile.set_defaults(func=_prepare_input)

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

    parser_eval = subparsers.add_parser('compute-evaluation',
                                        help='compute evaluation over original dataset using obtained representations')
    parser_eval.add_argument("-d", "--data", required=True, help="path to original dataset")
    parser_eval.add_argument("-g", "--generated_data", required=True, help="path to output file")
    parser_eval.add_argument("-o", "--output-dir", help="path to output dir")
    parser_eval.add_argument("-m", "--map", help="path to mapping file")
    parser_eval.add_argument("-t", "--type", required=True, choices=["ks", "dtfit", "tfit_mit"], help="dataset type")
    parser_eval.add_argument("-v", "--vectors", help='path to file containing vectors')

    parser_eval.set_defaults(func=_compute_evaluation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
