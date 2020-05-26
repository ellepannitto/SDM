import logging
import string

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

    # print("[CONLL READER] - ", filepath)

    accepted_chars = string.ascii_letters + "01234567890.-'"
    BASIC_FIELD_TO_IDX = {'id', 'text', 'lemma', 'upos', 'head', 'deprel'}

    with open(filepath) as fin:
        sentence = []
        skip_sentence = False

        for line in fin:
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

                    for head_plus_rel in deprels:
                        try:
                            rel_label, head = head_plus_rel.rsplit("=", 1)
                            head = int(head)

                            token_dict = {'id': id, 'text': text, 'lemma': lemma,
                                          'upos': upos, 'head': head, 'deprel': rel_label}
                            sentence.append(token_dict)
                        except:
                            skip_sentence = True
                            print("ILL FORMED LINE", line)

        if len(sentence):
            if not skip_sentence:
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


def DependencyBuilder(sentence):
    d = {}
    deps = {}
    for tok in sentence:
        ide = int(tok["id"])
        d[ide] = tok
        head = int(tok["head"])
        if head not in deps:
            deps[head] = []
        deps[head].append((ide, tok["deprel"]))

        if head == 0:
            d[0] = tok
    yield d, deps

if __name__ == "__main__":
    import glob
    folder = "/home/ludovica/corpus/ukwac1/"

    for file in glob.glob(folder+"*"):
        for el in CoNLLReader(file, delimiter=" "):
            print(el)
            input()