import copy
import logging
import string
import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)


def StanzaReader(text, preprocessing_fn=None, nlp=None):
    if not nlp:
        import stanza
        nlp=stanza.Pipeline('en')
    if preprocessing_fn:
        text = preprocessing_fn(text)
    doc = nlp(text)
    for sentence in doc.sentences:
        yield sentence


def CoNLLReader(delimiter, filepath):
    """

    :param str delimiter: character that separates CONLL file's columns (" " or "\t")
    :param str filepath: path to input CONLL file
    :return: a sentence in the form of a list of dictionaries, each dictionary is a token with the following keys: 'id', 'text', 'lemma', 'upos', 'head', 'deprel'
    :rtype: list[dict]
    """

    # print("[CONLL READER] - ", filepath)
    # logger.info("CONLL READER parsing {}".format(filepath))

    accepted_chars = string.ascii_letters + "01234567890.-'"
    BASIC_FIELD_TO_IDX = {'id', 'text', 'lemma', 'upos', 'head', 'deprel'}

    with open(filepath) as fin:
        sentence = []
        skip_sentence = False

        for line in fin:
        # for line in tqdm.tqdm(fin, desc="CoNLLReader"):
            line = line.strip()

            if not len(line):
                if len(sentence):
                    if not skip_sentence:
                        yield sentence
                    sentence = []
                    skip_sentence = False
            else:
                linesplit = line.split(delimiter)

                id, text, lemma, upos, ne, deprels = linesplit
                deprels = deprels.split(",")

                if any(c in string.ascii_letters for c in lemma) and all(c in accepted_chars for c in lemma):

                    if upos in ["NNP", "NNPS"] and not ne == "O":
                        text = ne
                        lemma = ne
                    else:
                        lemma = lemma.lower()

                    for head_plus_rel in deprels:
                        try:
                            rel_label, head = head_plus_rel.rsplit("=", 1)
                            #head = int(head)

                            token_dict = {'id': id, 'text': text, 'lemma': lemma,
                                          'upos': upos, 'head': head, 'deprel': rel_label}
                            sentence.append(token_dict)
                        except:
                            skip_sentence = True
                            logger.info("ILL FORMED LINE:{} (in {})".format(line,filepath))

        if len(sentence):
           if not skip_sentence:
               yield sentence

    logger.info("Finish reading: {}".format(filepath))


def ukWaCReader(filepath):

    with open(filepath) as fin:
        sentence = []

        for line in fin:
            line = line.strip()
            if not len(line):
                if len(sentence):
                    yield sentence
                    sentence = []
            else:
                sentence.append(line)

        if len(sentence):
            yield sentence

def DependencyBuilder(accepted_pos, accepted_rel, sentence, refine=True):
    """
    :param list accepted_pos: filter out lemmas with pos out of this list
    :param list accepted_rel: filter out relations out of that list
    :param list sentence: a list of dictionaries, each dictionary represent a tokens with the following keys: 'id', 'text', 'lemma', 'upos', 'head', 'deprel'
    :param boolean refine: a flag that indicates if applying morpho-syntactic refinements or not
    :returns:
            -words_dict (:py:class:`dict`) - a words dictionary {token_id : {lemma,upos}}
            -deps_ids_dict_copy () - a dependencies dictionary {head_id:[(dep_id, role)]}

    """
    # logger.info("Processing sentence: {}".format(sentence))
    def relation_standardization(role):
        if role == "nsubjpass":
            role = "dobj"
        elif role == "csubjpass":
            role = "ccomp"
        elif role == "nsubj:xsubj":
            role = "nsubj"
        elif role == "nmod:agent":
            role = "nsubj"
        return role

    def pos_standardization(pos):
        if pos.startswith("V"):
            pos = "V"
        elif pos.startswith("N"):
            pos = "N"
        elif pos.startswith("J"):
            pos = "J"
        return pos

    def refine_lemmas(w_dict, deps_dict):
        # refine lemmas
        for head_id in deps_dict:
            for dep_tup in deps_dict[head_id]:
                # 1: phrasal verbs
                if "prt" in dep_tup[1]:
                    try:
                        if w_dict[dep_tup[0]]["upos"] == "RP" and w_dict[head_id]["upos"] == "VERB":
                            w_dict[head_id]["lemma"] = "{}_{}".format(w_dict[head_id]["lemma"],
                                                                      w_dict[dep_tup[0]]["lemma"])
                    except KeyError:
                        pass
        # 2: personal pronouns
        for w_id in w_dict.keys():
            word = w_dict[w_id]
            if word["lemma"] in ["I", "he", "she", "you", "we", "they"]:
                word["lemma"] = "PERSON"
                word["upos"] = "N"

    # read input sentence
    words_dict = {} # {token_id : {lemma,upos}}
    deps_ids_dict = defaultdict(list) # {head_id:[(dep_id, role)]}
    for token in sentence:
        pos = pos_standardization(token["upos"])
        # take only words with a given PoS
        if pos in accepted_pos:
            # take only dependencies with a given label
            role = relation_standardization(token["deprel"])
            if role.startswith(tuple(accepted_rel)):
                deps_ids_dict[token["head"]].append((token["id"], role))
            if token["id"] not in words_dict:
                words_dict[token["id"]] = {'lemma': token["lemma"], 'upos': pos}

    if refine: refine_lemmas(words_dict, deps_ids_dict)

    # filter
    deps_ids_dict_res = defaultdict(list)
    for h_id, deps in deps_ids_dict.items():
        for dep in deps:
            try:
                deps_ids_dict_res[h_id].append(dep)
            except KeyError:
                pass


    yield words_dict, deps_ids_dict_res

if __name__ == "__main__":
    import glob
    folder = "/home/ludovica/corpus/ukwac1/"

    for file in glob.glob(folder+"*"):
        for el in CoNLLReader(file, delimiter=" "):
            print(el)
            input()