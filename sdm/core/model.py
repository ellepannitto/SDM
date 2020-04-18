import sdm.utils.data_utils as dutils

import random


class LinguisticConditions:
    def __init__(self, relations, vectors):
        self.content = {r:[] for r in relations}
        self.vectors = vectors

    def add(self, rel, form):
        self.content[rel].append(form)

class ActiveContext:
    def __init__(self, relations, vectors):
        self.content = {r:[] for r in relations}
        self.vectors = vectors

    def integrate(self, GEK_portion, n=20, weight_forward = True, weight_backward = False, weight_measure = 'cosine'):

        for relation in self.content:

            head_content = self.get_representation(self.content[relation], n)
            head_GEK = self.get_representation(GEK_portion[relation], n)

            # TODO: what to do with empty vector?

            #weight forward
            #weight backward
            #update

    def get_representation(self, box, n):

        centroid = None
        for sub_box in box:
            for el in box[:n]:
                form, pos, w = el


class StructuredDistributionalModel:
    def __init__(self, graph, relations_map, relations_list, vectors):
        self.graph = graph.session()
        self.rel_map = relations_map

        self.vectors = vectors

        self.relations = relations_list
        self.LC = LinguisticConditions(self.relations, self.vectors)
        self.AC = ActiveContext(self.relations, self.vectors)

    def process(self, form, pos, rel, K):

        print("processing", form, pos, rel)

        self.LC.add(rel, form)
        GEK = self.extract_GEK(form, rel, pos, K)
        self.AC.integrate(GEK)



    def extract_GEK(self, form, rel, pos, K):
        GEK = {}
        for box_relation in self.rel_map:
            print("box: ", box_relation)

            GEK[box_relation] = []
            if box_relation == rel:
                GEK[box_relation].append((form, pos, 1))
                # GEK[box_relation].append((form, pos, self.vectors[form], 1))
            else:
                none_in_list_in = 'None' in self.rel_map[rel]
                list_in_wo_none = [x for x in self.rel_map[rel] if not x=='None']

                none_in_list_out = 'None' in self.rel_map[box_relation]
                list_out_wo_none = [x for x in self.rel_map[box_relation] if not x=='None']

                query_str_prefix = "MATCH (n:words {form:$form, POS:$pos}) - [a:args] - (e:events) - [a2:args] - (m:words) "
                query_str_middle = "WHERE NOT m.form IN ['LOCATION', 'PERSON', 'ORGANIZATION'] "
                # query_str_suffix = " RETURN n.form, a.role, e.form, a2.role, a2.pmi, m.form, m.POS " \
                query_str_suffix = " RETURN n.form, a.role, a2.role, sum(a2.pmi) AS PMI, m.form, m.POS " \
                                    "ORDER BY PMI DESC " \
                                    "LIMIT $K"

                if none_in_list_in:
                    query_str_middle+="AND (a.role in $forms_in OR a.role is null) "
                else:
                    query_str_middle += "AND (a.role in $forms_in) "

                if none_in_list_out:
                    query_str_middle+="AND (a2.role in $forms_out OR a2.role is null)"
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
                                      K=K)

                for el in data.data():
                    GEK[box_relation].append((el["m.form"], el["m.POS"], el["PMI"]))
                    # print(GEK[box_relation])
                    # input()

        return GEK


def build_representation(output_path, graph, relations_fpath, data_fpaths, vector_fpath):

    relations_map = dutils.load_mapping(relations_fpath)

    vectors = {}
    vectors = dutils.load_vectors(vector_fpath, len_vectors=10)

    for filename in data_fpaths:

        dataset = dutils.load_dataset(filename)
        for item in dataset:
            elements, object_relation, ac_content = item
            sdm = StructuredDistributionalModel(graph, relations_map, ac_content, vectors)
            for word in elements:
                form, pos, rel = word

                sdm.process(form, pos, rel, 50)