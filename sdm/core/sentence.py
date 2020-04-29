import re, sys, string
from collections import defaultdict


class Sentence(object):
    """
    Sentence class, used to represent sentences.
    A property, which corresponds to a dictionary of words (Word class).
    Inside the dictionary, each word is accessed by a key (word's ID).
    """

    def __init__(self, pos=None, roles=None):
        self._dictionary = {}
        if roles is None:
            roles = ["nsubj", "dobj", "iobj", "nmod", "amod", "compound"]
        if pos is None:
            pos = [("V", "N"), ("N", "J"), ("N", "N")]
        self.acceptable_pos_patterns = pos
        # ("N", "N"), ("V", "N"), ("N", "J"), ("J", "N"))
        self.accettable_roles = roles

    # ["nsubj", "dobj", "iobj", "nmod", "appos", "vocative"]
    # The pipeline refers to Universal Dependencies v1
    # https://universaldependencies.org/docsv1/u/dep/index.html

    def add(self, tup):
        """
        Add word to sentence dictionary.
        :param tup: tuple
        """
        word = Word(tup)
        word_id = word.get_index()
        self._dictionary[word_id] = word

    def give_dictionary(self):
        """
        Access sentence dictionary.
        :return: sentence dictionary
        """
        return self._dictionary
    """
    REMEMBER: dictionaries / lists that are properties of a object CANNOT be accessed from outside.
    If you want to access them, use the appropriate method 'give_dictionary()'.
    """

    def give_keys(self):
        """
        Return sorted sentence dictionary's keys.
        :return: list
        """
        return sorted(self._dictionary.keys())

    # Methods to return Sentence's words information

    def return_sentence(self):
        """
        Return the original sentence (with annotation).
        :return: a string
        """
        s = ""
        for key in self.give_keys():
            s += " ".join(self._dictionary[key].return_word()) + "\n"
        s += "\n"
        return s

    def return_refined_sentence(self):
        """
        Return the refined sentence (with annotation) taking only relevant POS.
        :return: a string
        """
        s = ""
        for key in self.give_keys():
            if self._dictionary[key].get_cpos():
                s += " ".join(self._dictionary[key].return_refined_word()) + "\n"
        s += "\n"
        return s

    def return_tokens(self):
        """
        Return the original sentence (without annotation).
        :return: a string
        """
        s = ""
        for key in self.give_keys():
            s += " ".join(self._dictionary[key].return_token()) + "\n"
        return s

    # FROM GEK: rules to refine tokens properties
    def refine_tokens(self):
        # loop 1: refine lemmas and POS
        for key in self.give_keys():

            token = self._dictionary[key]
            lemma = token.get_lemma().lower()
            # filter words that contains non alfanumerical characters
            admitted_chars = string.ascii_letters + " .-0123456789"
            if all(c in admitted_chars for c in lemma):
                """
                # 1. Multiwords
                # a. MWE
                tmp = token.get_deps()
                for role, head_index in token.get_deps():
                    if "compound" in role:

                        #1	The	the	DT	O	3	det
                        #2	basketball	basketball	NN	O	compound=3
                        #3	match	match	NN	O	5	nsubjpass
                        #4	is	be	VBZ	O	5	auxpass
                        #5	canceled	cancel	VBN	O	0	ROOT
                        #6	.	.	.	O	_	_

                        #Refinement gives the following result

                        #2	basketball	basketball	_	O	
                        #3	match	basketball_match	NN	O	5	nsubjpass


                        # retrieve the head
                        head_token = self._dictionary[head_index]
                        if head_token.get_pos()[0] == "N":
                            if head_index == key+1:
                                head_token._improved_lemma = token._lemma + "_" + head_token._lemma
                                tmp.remove((role, head_index))
                            elif head_index == key-1:
                                head_token._improved_lemma = head_token._lemma + "_" + token._lemma
                                tmp.remove((role, head_index))
                            else:
                                tmp.remove((role, head_index))
                            token._pos = "_"


                token._dependencies = tmp
                del tmp
                """

                # 2. Coarse PoS
                # a. if this is a PROPER NOUNS
                if token.get_pos() in ("NNP", "NNPS"):
                    if token.get_ne() in (
                    "PERSON", "ORGANIZATION", "LOCATION", "COUNTRY", "CITY", "STATE_OR_PROVINCE", "TIME"):
                        token._improved_lemma = token.get_ne()

                # b. assign coarse pos to Nouns, Verbs, Adjectives
                token.assign_coarse_pos()

                # 3. Phrasal verbs
                # a. if this is a particle

                if token.get_pos() == "RP":

                    # for each dependence to a head
                    for child_role, head_index in token.get_deps():

                        # if the particle points to a verb through a enhanced dep
                        if "prt" in child_role:

                            # retrieve the head
                            head_token = self._dictionary[head_index]
                            # if this is a verb, improved its lemma

                            if head_token.get_pos()[0] == "V":
                                head_token._improved_lemma = head_token.get_lemma().lower() + "_" + lemma

                # b. if this is a personal pronoun
                if token.get_pos() == "PRP":
                    # if lemma corresponds to a human
                    if token.get_lemma() in ("I", "he", "she", "you", "we", "they"):
                        token._improved_lemma = "PERSON"
                        token._named_entity = "PERSON"
                        token._coarse_pos = "N"
                    else:
                        token.ID = None
            else:
                pass
        # loop 2: assign tokenID (lemma/POS)
        for key in self.give_keys():
            self._dictionary[key].assign_token_id()



class Word(object):
    """
	Word class, used to represent words of a sentence.
	Each Word corresponds to a tuple of values:
	- index
	- token
	- lemma
	- PoS
	- named entity
	- UD dependencies
	Information derived from refinements:
	- ID: the lemma + "/" + the coarse PoS
	- coarse_pos : coarse PoS ("V")
	- improved_lemma: the lemma refined by some rules
	Methods are used to return or set different tuple's elements.
	The ID of a word is converted in integer for simplicity.
	"""

    def __init__(self, tup):

        self._index = tup[0]
        self._token = tup[1]
        self._lemma = tup[2]
        self._pos = tup[3]
        self._named_entity = tup[4]

        try:
            # list of (role, head) tuples
            self._dependencies = [(i.split("=")[0], int(i.split("=")[1]))
                                  for i in tup[5].split(",") if len(i.split("=")) == 2]

        except AttributeError:
            pass
        # self.dependencies = dict([("root", 0)])
        except ValueError:
            # print(tup)
            self._dependencies = []

        # Refined, event worthy, type of data
        self._ID = None
        self._coarse_pos = None
        self._improved_lemma = None

    # Methods to return attribute values
    def get_index(self):
        return self._index

    def get_token(self):
        return self._token

    def get_lemma(self):
        return self._lemma

    def get_pos(self):
        return self._pos

    def get_ne(self):
        return self._named_entity

    def get_deps(self):
        return self._dependencies

    def get_cpos(self):
        return self._coarse_pos

    def get_token_id(self):
        return self._ID

    def get_improved_lemma(self):
        return self._improved_lemma