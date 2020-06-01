import os, sys
import subprocess
import logging
from shutil import copyfile
from neo4j import GraphDatabase
from sdm.utils import data_utils as dutils


logger = logging.getLogger(__name__)


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

    N_lemmas = dutils.count_absolute_freq(stats_path)
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
            freq = float(freq)
            prob = freq / N_lemmas
            line = '\t'.join([str(c), lemma, pos, str(int(freq)), str(prob)])
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
            freq = float(freq)
            prob = freq / n_events_dic[n]
            line = '\t'.join([str(c), event, str(int(freq)), str(prob), str(n)])
            writer.write(line + "\n")
            events_idx[event] = str(c)

    # write event-word edges
    output_file = os.path.join(output_path, "event-word_edges.csv")
    with open(output_file, 'w') as writer:
        header = '\t'.join([':START_ID(event-ID)', 'freq:int', 'prob:float', 'pmi:float', 'degree:int', 'role', ':END_ID(word-ID)'])
        writer.write(header + "\n")
        for c, event in enumerate(events):
            c = c + 1
            event, freq = event
            freq = int(float(freq))
            event_form = event.replace(" ", ",")
            for w in event.split():
                l,p,r = w.split("@")
                lemma_pos = "{}@{}".format(l,p)
                deg = len(event.split(" "))
                line = "\t".join([events_idx[event_form], str(freq), "0.0", "0.0", str(deg), r, lemmas_idx[lemma_pos]])
                writer.write(line + "\n")
                # TO DO: compute frequency


def import_graph(neo4j_folder, data_path, copy=True):
    # from Neo4j importation procedure of Patrick

    # Check Neo4j server status
    msg =subprocess.Popen(["bin/neo4j","status"],stdout=subprocess.PIPE, cwd=neo4j_folder)
    msg = msg.stdout.read()
    logger.info(msg.strip())
    if msg.strip() == b"Neo4j is not running":
        # Check whether there is a database
        if len(os.listdir(os.path.join(neo4j_folder, "data/databases"))) >0:
            # If there is one, remove it
            subprocess.Popen(["rm", "-fr", "*.*"], cwd=os.path.join(neo4j_folder, "data/databases"))

        files = os.listdir(data_path)
        # Copy data to import folder (optional)
        if copy:
            for f in files:
                copyfile(os.path.join(data_path,f), os.path.join(neo4j_folder, "import", f))

        # Import
        if len(files) == 3: # only s
            args = ["bin/neo4j-admin", "import",
                    "--nodes:events", "import/events_nodes.csv",
                    "--nodes:words", "import/words_nodes.csv",
                    "--relationships:args", "import/event-word_edges.csv",
                    "--delimiter", "TAB",
                    "--database", "graph.db",
                    "--ignore-missing-nodes",
                    "--ignore-duplicate-nodes"]
        else: #also a
            args = ["bin/neo4j-admin", "import",
                    "--nodes:events", "import/events_nodes.csv",
                    "--nodes:words", "import/words_nodes.csv",
                    "--relationships:args", "import/event-word_edges.csv",
                    "--relationships:asso", "import/word-word_edges.csv",
                    "--delimiter", "TAB",
                    "--database", "graph.db",
                    "--ignore-missing-nodes",
                    "--ignore-duplicate-nodes"]
        subprocess.Popen(args, cwd=neo4j_folder)
    else:
        logger.info(msg.strip(), "Close Neo4j using Ctr+C")
        sys.exit(2)
        # blocca ed esci
