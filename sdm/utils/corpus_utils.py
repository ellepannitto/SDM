import logging

logger = logging.getLogger(__name__)


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



if __name__ == "__main__":
    import glob
    folder = "/home/ludovica/corpus/ukwac1/"

    for file in glob.glob(folder+"*"):
        for el in ukWaCReader(file):
            print(el)
            input()