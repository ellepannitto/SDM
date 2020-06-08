"""
model.py the core architecture for Structured Distributional Model (SDM).
"""
import os
import logging
import numpy as np

import sdm.utils.data_utils as dutils
import sdm.utils.weight_utils as wutils
import sdm.utils.representation_utils as rutils

logger = logging.getLogger(__name__)


def rerank(weighted_list, head_vector, ranking_function):
    logger.info("list to re-rank: {}".format([x[0] for x in weighted_list]))

    new_list = ranking_function(weighted_list, head_vector)
    new_list.sort(key=lambda x: -x[2])

    logger.info("new list: {}".format([x[0] for x in new_list]))

    return new_list


class LinguisticConditions:
    def __init__(self, sdm):
        self.sdm = sdm
        self.content = {r: [] for r in self.sdm.relations}
        logger.info("Initializing LC: {}".format(self.content))

    def update(self, rel, vector, label):
        print("* previous LC:", file=self.sdm.log_file)
        for el in self.content:
            print("\t{} - {}".format(el, " ".join(x[0] for x in self.content[el])), file=self.sdm.log_file)

        self.content[rel].append((label, vector))

        print("* current LC:", file=self.sdm.log_file)
        for el in self.content:
            print("\t{} - {}".format(el, " ".join(x[0] for x in self.content[el])), file=self.sdm.log_file)

        logger.info("Updating LC: {}".format(self.content[rel]))

    def get_vector(self, target_relation):

        logger.info("Returning LC vector for relation {}".format(target_relation))

        if target_relation == 'SENTENCE':
            vectors = []
            for relation in self.content:
                if len(self.content[relation]) > 0:
                    vectors.append(self.content[relation][0][1])
            ret = np.sum(vectors, axis=0)
            return ret / len(vectors)
        elif len(self.content[target_relation]) > 0:
            ret = self.content[target_relation][0][1]
            return ret
        else:
            return None


class ActiveContext:
    def __init__(self, sdm):
        self.sdm = sdm
        self.content = {r: [] for r in self.sdm.relations}
        logger.info("Initializing AC: {}".format(self.content))

    def update(self, GEK_portion):

        print("* previous AC:", file=self.sdm.log_file)
        for relation in self.content:
            substr = ''
            for sublist in self.content[relation]:
                substr += ", ".join(x[0] for x in sublist)
                substr += " || "
            print("\t{} - {}".format(relation, substr), file=self.sdm.log_file)

        logger.info("Updating AC...")

        for relation in self.content:
            logger.info("Examining relation {}".format(relation))

            if self.sdm.rank_forward:
                print("* computing centroid of AC content for relation {}".format(relation))
                head_content = self.sdm.get_representation_function(self.content[relation], self.sdm.M)
                logger.info("Head_content computed: {}".format(head_content))

                if head_content is not None:
                    logger.info("Re-ranking GEK_portion with respect to head_content")
                    GEK_portion[relation] = rerank(GEK_portion[relation], head_content, self.sdm.weight_function)

            if self.sdm.rank_backward:
                print("* computing centroid of GEK portion content for relation {}".format(relation))
                head_GEK = self.sdm.get_representation_function([GEK_portion[relation]], self.sdm.M)
                logger.info("Head_GEK computed: {}".format(head_GEK))

                if head_GEK is not None:
                    logger.info("Re-ranking AC content with respect to head_GEK")
                    new_content = []
                    for sublist in self.content[relation]:
                        new_content.append(rerank(sublist, head_GEK, self.sdm.weight_function))
                    self.content[relation] = new_content

                    logger.info("new content for relation {}: "
                                "{}".format(relation, [[x[0] for x in sublist] for sublist in self.content[relation]]))

            logger.info("Appending GEK_portion to content for relation {}".format(relation))

            if len(GEK_portion[relation]) > 0:
                self.content[relation].append(GEK_portion[relation])
            else:
                logger.info("GEK portion empty")
                # input()

        print("* current AC:", file=self.sdm.log_file)
        for relation in self.content:
            substr = ''
            for sublist in self.content[relation]:
                substr += ", ".join(x[0] for x in sublist)
                substr += " || "
            print("\t{} - {}".format(relation, substr), file=self.sdm.log_file)

    def get_vector(self, target_relation):
        logger.info("Returning AC vector for relation {}".format(target_relation))

        if target_relation == 'SENTENCE':
            vectors = []
            for relation in self.content:
                ret = self.sdm.get_representation_function(self.content[relation], self.sdm.M)
                if ret is not None:
                    vectors.append(ret)
            ret = np.sum(vectors, axis=0)
            return ret / len(vectors)
        else:
            ret = self.sdm.get_representation_function(self.content[target_relation], self.sdm.M)
            return ret


class StructuredDistributionalModel:
    def __init__(self, graph, relations_map, vectors, weight_function, rank_forward, rank_backward,
                 N, M, include_same_relations, representation_function, weight_to_extract):

        logger.info("Initializing new structured distributional model")
        logger.info("setting SDM global parameters")  # TODO: add full dump of parameters

        self.graph = graph.session()
        self.rel_map = relations_map
        self.vector_space = vectors

        self.weight_function = weight_function
        self.get_representation_function = representation_function
        self.rank_forward = rank_forward
        self.rank_backward = rank_backward
        self.N = N
        self.M = M
        self.include_same_relations = include_same_relations

        self.weight_to_extract = weight_to_extract

        self.log_file = None
        self.relations = None
        self.LC, self.AC = None, None

    def new_item(self, relations_list, log_fout_handler):
        logger.info("Initializing new SDM item")

        self.log_file = log_fout_handler
        self.relations = relations_list
        print("RELATIONS: {}".format(self.relations), file=self.log_file)
        self.LC = LinguisticConditions(self)
        self.AC = ActiveContext(self)

    def process(self, form, pos, rel):

        logger.info("processing element {} - {} - {}".format(form, pos, rel))

        if form in self.vector_space:
            print("[STEP1: EXTRACTING GEK for {}@{}@{}]".format(form, pos, rel), file=self.log_file)
            GEK = self.extract_GEK(form, rel, pos)
            print("[STEP2: UPDATING LC for {}@{}@{}]".format(form, pos, rel), file=self.log_file)
            self.LC.update(rel, self.vector_space[(form, pos)], form)
            print("[STEP3: UPDATING AC for {}@{}@{}]".format(form, pos, rel), file=self.log_file)
            self.AC.update(GEK)
        else:
            logger.info("WARNING: element not in vector space")
            print("[WARNING]: element {} not in vector space".format(form), file=self.log_file)

    def extract_GEK(self, form, rel, pos):

        GEK = {}
        logger.info("GEK portion initialized: {}".format(GEK))

        for box_relation in self.relations:

            logger.info("extracting knowledge for label {}".format(box_relation))

            GEK[box_relation] = []
            if box_relation == rel:
                logger.info("label {} is equal to relation {}".format(box_relation, rel))
                if self.include_same_relations:
                    # TODO: do we normalize the PMI between 0 and 1 always?
                    GEK[box_relation].append((form, self.vector_space[(form, pos)], 1))
                    logger.info("Adding {} as GEK for label {}".format(form, box_relation))
            else:
                none_in_list_in = 'None' in self.rel_map[rel]
                list_in_wo_none = [x for x in self.rel_map[rel] if not x == 'None']

                none_in_list_out = 'None' in self.rel_map[box_relation]
                list_out_wo_none = [x for x in self.rel_map[box_relation] if not x == 'None']

                query_str_prefix = "MATCH (n:words {form:$form, POS:$pos}) - [a:args] - (e:events) - [a2:args] - (m:words) "
                query_str_middle = "WHERE NOT m.form IN ['LOCATION', 'PERSON', 'ORGANIZATION'] "
                query_str_suffix = ""

                if self.weight_to_extract == 'pmi':
                    query_str_suffix = " RETURN n.form, a.role, a2.role, sum(a2.pmi) AS PMI, m.form, m.POS " \
                                       " ORDER BY PMI DESC " \
                                       " LIMIT $K "
                elif self.weight_to_extract == "lmi":
                    query_str_suffix = " RETURN n.form, a.role, a2.role, sum(a2.pmi*a2.freq) AS PMI, m.form, m.POS " \
                                       " ORDER BY PMI DESC " \
                                       " LIMIT $K "

                if none_in_list_in:
                    query_str_middle += "AND (a.role in $forms_in OR a.role is null) "
                else:
                    query_str_middle += "AND (a.role in $forms_in) "

                if none_in_list_out:
                    query_str_middle += "AND (a2.role in $forms_out OR a2.role is null)"
                else:
                    query_str_middle += "AND (a2.role in $forms_out) "

                query = query_str_prefix+query_str_middle+query_str_suffix
                logger.info("performing query: {}".format(query))
                logger.info("PARAMETERS: form={}, pos={}, role={}, forms_in={}, "
                            "forms_out={}, K={}".format(form, pos, rel, list_in_wo_none, list_out_wo_none, self.N))
                # input()

                data = self.graph.run(query,
                                      form=form, pos=pos, role=rel,
                                      forms_in=list_in_wo_none, forms_out=list_out_wo_none,
                                      K=self.N)
                #for record in data: print(record.data())
                #for el in data.data():
                for el in data:
                    el = el.data()
                    # TODO: how to have N words if something is not in vector space?
                    el_form = el["m.form"]
                    el_pos = el["m.POS"]
                    logger.info("extracted word {}".format(el_form))
                    if el_form in self.vector_space:
                        el_form_v = self.vector_space[(el_form, el_pos)]
                        pmi = el["PMI"]
                        GEK[box_relation].append((el_form, el_form_v, pmi))
                    else:
                        logger.info("word not in vector space")

            logger.info("GEK for label {}: {}".format(box_relation, [(x[0], x[2]) for x in GEK[box_relation]]))
            print("* GEK for label {}:".format(box_relation), file=self.log_file)
            print("\t"+", ".join(x[0] for x in GEK[box_relation]), file=self.log_file)

        return GEK


    def get_vector(self, target_relation):

        LC_vector = self.LC.get_vector(target_relation)
        AC_vector = self.AC.get_vector(target_relation)
        return LC_vector, AC_vector


def build_representation(output_path, graph, relations_fpath, data_fpaths, vector_space,
                         weight_function, rank_forward, rank_backward, N, M,
                         include_same_relations, representation_function, weight_to_extract):

    f_weight_function = wutils.possible_functions[weight_function]
    f_representation_function = rutils.possible_functions[representation_function]

    relations_map = dutils.load_mapping(relations_fpath)
    vectors = vector_space

    sdm = StructuredDistributionalModel(graph=graph, relations_map=relations_map, vectors=vectors,
                                        weight_function=f_weight_function, rank_forward=rank_forward,
                                        rank_backward=rank_backward, N=N, M=M,
                                        include_same_relations=include_same_relations,
                                        representation_function=f_representation_function,
                                        weight_to_extract=weight_to_extract)

    for filename in data_fpaths:
        log_fname = output_path + os.path.basename(filename) + ".log"

        with open(log_fname, "w", buffering=1) as log_fout:
            out_fname = output_path+os.path.basename(filename)+".out"
            dataset = dutils.load_dataset(filename)
            res = []
            for item in dataset:
                logger.info("Processing dataset item: {}".format(item))
                print("## Processing dataset item: {} ##".format(item), file=log_fout)
                elements, object_relation, ac_content = item
                sdm.new_item(ac_content, log_fout)

                for word in elements:

                    form, pos, rel = word
                    sdm.process(form, pos, rel)

                print("Printing vectors in output", file=log_fout)
                lc_vector, ac_vector = sdm.get_vector(object_relation)
                res.append((lc_vector, ac_vector))

        dutils.dump_results(filename, res, out_fname)

if __name__ == '__main__':
    from sdm.utils import graph_utils as gutils
    graph=gutils.connect_to_graph("bolt://127.0.0.1:7687", "neo4j", "neo4j")
    v=dutils.load_vectors('/home/giulia/CORPORA/fastText-SGNS-w10-d300.txt', withPoS=False,
                        len_vectors=100)
    build_representation(output_path='data/lme/results', graph=graph, relations_fpath='home/giulia/PhD_projects/SDM/data/lme/metonymy.relations',
                         data_fpaths=['data/lme/generated/traxler.head_verbs_args'], vector_space=v,
                         weight_function='cosine', rank_forward=True, rank_backward=True,
                         N=50, M=20, include_same_relations=False,
                         representation_function='centroid',
                         weight_to_extract='pmi')