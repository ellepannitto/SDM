from neo4j import GraphDatabase


def extract_relations_list(output_path, graph):
    g = graph.session()

    query_possible_relations = g.run("MATCH (n)-[a:args]-(m) RETURN DISTINCT a.role")
    with open(output_path+"relations_list.txt", "w") as fout:
        for el in query_possible_relations.data():
            rel = el["a.role"]
            if rel is not None:
                print(rel, file=fout)

def connect_to_graph(uri, user, password):

    driver = GraphDatabase.driver(uri, auth=(user, password))

    return driver

# session = driver.session()

# print(session.run("MATCH (n:events) RETURN n LIMIT 10").data())

# print(session.run("MATCH (w:words) {form:'murder', POS:'N'} - [a:args {role: 'nsubj'}] - (e:events)  "
#                   "RETURN w.form, a.role, e.form ORDER BY a.pmi DESC LIMIT 10" ))


# S = session.run("MATCH (w:words {form:'murder', POS:'N'}) - [a:args {role: 'dobj'}] - (e:events) - [a2:args] - (n:words) "
#                   "RETURN w.form, a.role, a.pmi, e.form, a2.role, n.form ORDER BY a.pmi DESC LIMIT 10" )

# for el in S.data():
#     print("{}, {}, {}, {}, {}, {}".format(el["w.form"], el["a.role"], el["a.pmi"], el["e.form"], el["a2.role"], el["n.form"]))
