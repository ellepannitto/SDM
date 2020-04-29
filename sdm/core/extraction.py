from sdm.core.sentence import Sentence



def extract_events(sent,pos, roles):
    """
    Extract events from a sentence

    :param sent: a list of tokens's info
    :param pos: a list of acceptable PoS (V,N) ecc
    :param roles: a list of acceptable roles
    :return:
    """
    out_dir = ""


    sentence = Sentence(pos, roles)
    for token in sent:
        tokensplit = token.split()
        try:
            id, form, lemma, pos, ne, synrels = tokensplit
            synrels = synrels.split(",")
            split_synrels = []
            for rel in synrels:
                label, head = rel.split("=")
                split_synrels.append((label, head))
            tup = (int(id), form, lemma, pos, ne, split_synrels)
            sentence.add(tup)

        except:
            print(token)
