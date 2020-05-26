import copy
import logging
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


def CoNLLReader(filepath, delimiter=" "):
    BASIC_FIELD_TO_IDX = {'id', 'text', 'lemma', 'upos', 'head', 'deprel'}

    with open(filepath) as fin:
        sentence = []

        for line in fin:
            line = line.strip()
            print(line)
            ss
            if not len(line):
                if len(sentence):
                    yield sentence
                    sentence = []
            else:
                linesplit = line.split(delimiter)
                id, text, lemma, upos, ne, deprels = linesplit
                deprels = deprels.split(",")

                # TODO: check if text is admitted (special chars etc...)
                # TODO: check if substitution with NE is needed

                for head_plus_rel in deprels:
                    rel_label, head = head_plus_rel.split("=")
                    head = int(head)

                    # TODO: add only chosen deprels

                    token_dict = {'id': id, 'text': text, 'lemma': lemma,
                                  'upos': upos, 'head': head, 'deprel': rel_label}
                    sentence.append(token_dict)

        if len(sentence):
            yield sentence


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

def DependencyBuilder(sentence, accepted_pos, accepted_rel, refine=True):

    """
    :param sentence: a list of dictionaries, each dictionary represent a tokens with the following keys:
    'id', 'text', 'lemma', 'upos', 'head', 'deprel'
    Take care:
    1) there may be a token who is dependent from another not attested in the list -> pass
    2) in case of enhanced deps, a token is repeated more times in the list, one per relation
    """
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
                    if w_dict[dep_tup[0]]["upos"] == "RP" and w_dict[head_id]["upos"] == "VERB":
                        w_dict[head_id]["lemma"] = "{}_{}".format(w_dict[head_id]["lemma"],
                                                                      w_dict[dep_tup[0]]["lemma"])
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
        deps_ids_dict[token["head"]].append((token["id"], relation_standardization(token["deprel"])))
        if token["id"] not in words_dict:
            words_dict[token["id"]] = {'lemma': token["lemma"], 'upos':pos_standardization(token["upos"])}
    #if refine: refine_lemmas(words_dict, deps_ids_dict)

    # filter
    deps_ids_dict_copy = copy.deepcopy(deps_ids_dict)
    for h_id, deps in deps_ids_dict.items():
        if h_id not in words_dict.keys():
            del deps_ids_dict_copy[h_id]
        else:
            if words_dict[h_id]["upos"] in accepted_pos:
                for i, dep in enumerate(deps):
                    dep_id, rel = dep
                    if words_dict[dep_id]["upos"] in accepted_pos and rel in accepted_rel:
                        pass
                    else:
                        del words_dict[dep_id]
                        del deps_ids_dict_copy[h_id][i]
            else:
                del words_dict[h_id]
                del deps_ids_dict_copy[h_id]
            if len(deps_ids_dict_copy[h_id]) == 0: del deps_ids_dict_copy[h_id]
    yield words_dict, deps_ids_dict_copy


if __name__ == "__main__":
    import glob
    folder = "/home/ludovica/corpus/ukwac1/"

    for file in glob.glob(folder+"*"):
        for el in CoNLLReader(file):
            print(el)
            input()