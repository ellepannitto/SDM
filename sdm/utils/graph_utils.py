import os
from neo4j import GraphDatabase
from sdm.utils import data_utils as dutils

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


def write_graph(stats_path, events_path, n_events_path, output_path):
    lemmas = dutils.load_freqs(stats_path)
    events = dutils.load_freqs(events_path)

    lemmas_idx = {}
    events_idx = {}

    n = dutils.count_absolute_freq(stats_path)
    n_events_dic = dutils.load_n_events_freq(n_events_path)

    # write word nodes
    output_file = os.path.join(output_path, "words_nodes.csv")
    with open(output_file, 'w') as writer:
        header = '\t'.join(['wordId:ID(word-ID)', 'form', 'POS', 'freq:int', 'prob:float'])
        writer.write(header + "\n")
        for c, lemma in enumerate(lemmas):
            c=c+1
            lemma_pos, freq = lemma
            lemma, pos = lemma_pos.split(" ")
            prob = int(freq)/ n
            line = '\t'.join([str(c), lemma, pos, freq, str(prob)])
            writer.write(line + "\n")
            lemmas_idx["{}@{}".format(lemma, pos)] = str(c)

    # write event nodes
    output_file = os.path.join(output_path, "events_nodes.csv")
    with open(output_file, 'w') as writer:
        header = '\t'.join(['eventId:ID(event-ID)', 'form', 'freq:int', 'prob:float', 'deg:int'])
        writer.write(header + "\n")
        for c, event in enumerate(events):
            c=c+1
            event, freq = event
            n = len(event.split(" "))
            event = event.replace(" ", ",")
            prob = int(freq) / n_events_dic[n]
            line = '\t'.join([str(c), event, freq, str(prob), str(n)])
            writer.write(line + "\n")
            events_idx[event] = str(c)

    # write event-word edges
    output_file = os.path.join(output_path, "event-word_edges.csv")
    with open(output_file, 'w') as writer:
        header = '\t'.join([':START_ID(event-ID)', 'freq:int', 'prob:float', 'pmi:float', 'degree:int', 'role' ':END_ID(word-ID)'])
        writer.write(header + "\n")
        for c, event in enumerate(events):
            c = c + 1
            event, freq = event
            event_form = event.replace(" ", ",")
            for w in event.split():
                l,p,r = w.split("@")
                lemma_pos = "{}@{}".format(l,p)
                deg = len(event.split(" "))
                line = "\t".join([events_idx[event_form], str(freq), "_", "_", str(deg), r, lemmas_idx[lemma_pos]])
                writer.write(line + "\n")
                # TO DO: compute frequency


if __name__ == '__main__':
    fold = "/home/giulia/PhD_projects/results-gek"
    s = os.path.join(fold, "lemma-freqs.txt")
    e = os.path.join(fold, "events-freqs.txt")
    write_graph(s,e,fold)