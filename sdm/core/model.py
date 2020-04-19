import numpy as np

import sdm.utils.data_utils as dutils
import sdm.utils.weight_utils as wutils
import sdm.utils.representation_utils as rutils


class LinguisticConditions:
    def __init__(self, sdm):
        self.sdm = sdm
        self.content = {r: [] for r in self.sdm.relations}

    def update(self, rel, vector, label):
        self.content[rel].append((label, vector))


class ActiveContext:
    def __init__(self, sdm):
        self.sdm =  sdm
        self.content = {r: [] for r in self.sdm.relations}

    def update(self, rel, GEK_portion):

        for relation in self.content:
            head_content = self.sdm.get_representation_function(self.content[relation], self.sdm.M)
            head_GEK = self.sdm.get_representation_function([GEK_portion[relation]], self.sdm.M)

            if not head_content == 0:
                if self.sdm.rank_forward:
                    GEK_portion[relation] = self.rerank(GEK_portion[relation], head_content, self.sdm.weight_function)

            if not head_GEK == 0:
                if self.sdm.rank_backwards:
                    new_content = []
                    for sublist in self.content[relation]:
                        new_content.append(self.rerank(sublist, head_GEK, self.sdm.weight_function))
                    self.content[relation] = new_content

            self.content[relation].append(GEK_portion[relation])


    def rerank(self, head_vector, weighted_list, ranking_function):

        new_list = ranking_function(weighted_list, head_vector)
        new_list.sort(key=lambda x: -x[2])

        return new_list


class StructuredDistributionalModel:
    def __init__(self):
        pass

    def set_parameters(self, graph, relations_map, vectors, weight_function, rank_forward, rank_backward,
                 N, M, include_same_relations, representation_function):
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

    def new_item(self, relations_list):
        self.relations = relations_list
        self.LC = LinguisticConditions(self)
        self.AC = ActiveContext(self)

    def process(self, form, pos, rel, K):

        print("processing", form, pos, rel)
        if form in self.vector_space:

            GEK = self.extract_GEK(form, rel, pos, K)

            self.LC.update(rel, self.vector_space[form], form)
            self.AC.update(rel, GEK)

        else:
            print("word not in vector space", form)

    def extract_GEK(self, form, rel, pos):

        GEK = {}

        for box_relation in self.rel_map:
            print("box: ", box_relation)

            GEK[box_relation] = []
            if box_relation == rel:
                if self.include_same_relations:
                    # TODO: do we normalize the PMI between 0 and 1 always?
                    GEK[box_relation].append((form, self.vector_space[form], 1))
            else:
                none_in_list_in = 'None' in self.rel_map[rel]
                list_in_wo_none = [x for x in self.rel_map[rel] if not x == 'None']

                none_in_list_out = 'None' in self.rel_map[box_relation]
                list_out_wo_none = [x for x in self.rel_map[box_relation] if not x == 'None']

                query_str_prefix = "MATCH (n:words {form:$form, POS:$pos}) - [a:args] - (e:events) - [a2:args] - (m:words) "
                query_str_middle = "WHERE NOT m.form IN ['LOCATION', 'PERSON', 'ORGANIZATION'] "
                query_str_suffix = " RETURN n.form, a.role, a2.role, sum(a2.pmi) AS PMI, m.form, m.POS " \
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

                # NOT EXISTS(a.role) OPPURE a.role is null
                #     query = "MATCH (n:words {form:$form, POS:$pos}) - [a:args] - (e:events) - [a2:args] - (m:words) " \
                #             "WHERE a.role IN $roles_in AND a2.role IN $roles_out" \
                #             "RETURN n.form, a.role, e.form, a2.role, a2.pmi, m.form, m.POS " \
                #             "ORDER BY a2.pmi DESC " \
                #             "LIMIT 10"

                query = query_str_prefix+query_str_middle+query_str_suffix
                print(query)
                data = self.graph.run(query,
                                      form=form, pos=pos, role=rel,
                                      forms_in=list_in_wo_none, forms_out=list_out_wo_none,
                                      K=self.N)

                for el in data.data():
                    form = el["m.form"]
                    if form in self.vector_space:
                        form_v = self.vector_space[form]
                        pmi = el["PMI"]

                        GEK[box_relation].append((form, form_v, pmi))
                    else:
                        print("form not in vectors", form)

        return GEK


def build_representation(output_path, graph, relations_fpath, data_fpaths, vector_fpath,
                         weight_function='cosine', rank_forward=True, rank_backward=False,
                         N=50, M=20, include_same_relations=True, representation_function='centroid'):

    f_weight_function = wutils.possible_functions[weight_function]
    f_representation_function = rutils.possible_functions[representation_function]

    relations_map = dutils.load_mapping(relations_fpath)
    vectors = dutils.load_vectors(vector_fpath, len_vectors=10)

    sdm = StructuredDistributionalModel()
    sdm.set_parameters(graph=graph, relations_map=relations_map, vectors=vectors,
                       weight_function=f_weight_function, rank_forward=rank_forward,
                       rank_backward=rank_backward, N=N, M=M,
                       include_same_relations=include_same_relations,
                       representation_function=f_representation_function)

    for filename in data_fpaths:

        out_fname = output_path+filename+".out"
        dataset = dutils.load_dataset(filename)
        res = []
        for item in dataset:
            elements, object_relation, ac_content = item
            sdm.new_item(ac_content)

            for word in elements:
                form, pos, rel = word

                sdm.process(form, pos, rel, 50)

            res.append(sdm.get_representation(object_relation))

        dutils.dump_results(filename, res, out_fname)
