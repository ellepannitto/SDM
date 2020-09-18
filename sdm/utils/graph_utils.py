import os
import sys
import tqdm
import itertools
import subprocess
import logging
import collections
import math
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


def write_graph(stats_path, events_path, output_path):
    lemmas = dutils.load_freqs_generator(stats_path)
    events = dutils.load_freqs_generator(events_path)

    lemmas_idx = {}
    events_involving_lemma = collections.defaultdict(lambda: collections.defaultdict(int))
    events_with_deg = collections.defaultdict(int)

    N_lemmas = dutils.count_absolute_freq(stats_path)
    # n_events_dic = dutils.load_n_events_freq(events_path)

    # write word nodes
    output_file = os.path.join(output_path, "words_nodes.csv")
    with open(output_file, 'w') as writer:
        header = '\t'.join(['wordId:ID(word-ID)', 'form', 'POS', 'freq:int', 'prob:float'])
        print(header, file=writer)
        for c, lemma in tqdm.tqdm(enumerate(lemmas), desc="Loading LEMMAS"):
            c = c+1
            lemma_pos, freq = lemma
            lemma, pos = lemma_pos.split(" ")
            freq = float(freq)
            prob = freq / N_lemmas
            line = '\t'.join([str(c), lemma, pos, str(int(freq)), str(prob)])
            print(line, file=writer)
            lemmas_idx["{}@{}".format(lemma, pos)] = str(c)

    for c, event in tqdm.tqdm(enumerate(events), desc="Counting EVENTS"):
        c = c + 1
        event, freq = event
        freq = float(freq)
        event_form = event.replace(" ", ",")
        event_split = event.split(" ")
        deg = len(event_split)

        events_with_deg[deg] += freq

        try:
            event_split_lemmapos = []
            for w in event_split:
                l, p, r = w.split("@")
                lemma_pos = "{}@{}".format(l, p)
                event_split_lemmapos.append(lemma_pos)
                events_involving_lemma[lemma_pos][deg] += freq
        except:
            print("issue with event: {}".format(event_split))

        if deg > 2:
            for i in range(2, len(event_split_lemmapos)):
                subsets = itertools.combinations(event_split_lemmapos, i)
                for el in subsets:
                    events_involving_lemma[" ".join(el)][deg] += freq

    # write event nodes
    # write event-word edges
    output_file_nodes = os.path.join(output_path, "events_nodes.csv")
    output_file_edges = os.path.join(output_path, "event-word_edges.csv")
    events = dutils.load_freqs_generator(events_path)

    with open(output_file_nodes, 'w') as writer_nodes, \
            open(output_file_edges, "w") as writer_edges:

        header = '\t'.join(['eventId:ID(event-ID)', 'form', 'freq:int', 'prob:float', 'deg:int'])
        print(header, file=writer_nodes)

        header = '\t'.join([':START_ID(event-ID)', 'freq:int', 'prob:float', 'condprob_EgW:float',
                            'condprob_WgE:float', 'pmi:float', 'degree:int', 'role', ':END_ID(word-ID)'])
        print(header, file=writer_edges)

        for c, event in tqdm.tqdm(enumerate(events), desc="processing EVENTS"):
            c = c + 1
            event, freq = event
            freq = float(freq)
            event_form = event.replace(" ", ",")
            event_split = event.split(" ")
            deg = len(event_split)
            prob = freq / events_with_deg[deg]

            line = '\t'.join([str(c), event_form, str(int(freq)), str(prob), str(deg)])
            print(line, file=writer_nodes)

            try:
                event_split_lemmapos = []
                for w in event_split:
                    l, p, r = w.split("@")
                    lemma_pos = "{}@{}".format(l, p)
                    event_split_lemmapos.append(lemma_pos)

                for w_idx, lemma_pos in enumerate(event_split_lemmapos):

                    l, p, r = event_split[w_idx].split("@")

                    other_words_in_event = event_split_lemmapos[:w_idx] + event_split_lemmapos[w_idx+1:]

                    prob_E_g_w = freq / events_involving_lemma[lemma_pos][deg]
                    prob_w_g_E = freq / events_involving_lemma[" ".join(other_words_in_event)][deg]

                    observed_f = freq
                    expected_f = (events_involving_lemma[lemma_pos][deg] *
                                  events_involving_lemma[" ".join(other_words_in_event)][deg]) / \
                                  events_with_deg[deg]

                    pmi_w_t_E = freq * math.log(observed_f/expected_f)

                    line = "\t".join([str(c), str(int(freq)),
                                      "{:.5f}".format(prob),
                                      "{:.5f}".format(prob_E_g_w),
                                      "{:.5f}".format(prob_w_g_E),
                                      "{:.5f}".format(pmi_w_t_E),
                                      str(deg), r, lemmas_idx[lemma_pos]])
                    print(line, file=writer_edges)

            except:
                print("issue with event: {}".format(event_split))

                # TODO: compute frequency


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
