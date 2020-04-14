import sdm.utils.data_utils as dutils

def build_representation(output_path, graph, relations_fpath, data_fpaths):

    relations = dutils.load_mapping(relations_fpath)

    for filename in data_fpaths:
        items = dutils.load_dataset(filename)