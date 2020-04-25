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
