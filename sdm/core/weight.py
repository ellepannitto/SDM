import neo4j

from sdm.utils import graph_utils as gutils

graph = gutils.connect_to_graph("bolt://localhost:7687", "neo4j", "n4j").session()

query = "MATCH (n:event) RETURN n"
graph.run(query)