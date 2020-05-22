import logging

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

            if not len(line):
                if len(sentence):
                    yield sentence
                    sentence = []
            else:
                linesplit = line.split(" ")

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

def DependencyBuilder()


if __name__ == "__main__":
    import glob
    folder = "/home/ludovica/corpus/ukwac1/"

    for file in glob.glob(folder+"*"):
        for el in CoNLLReader(file):
            print(el)
            input()