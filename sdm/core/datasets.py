import os, sys
import pandas as pd

from sdm.utils import data_utils as dutils


class Datasets(object):

    def __init__(self, infile, outfolder, args_order):
        self.data = None
        self.datafile = infile
        self.name = os.path.basename(infile).split(".")[0]
        self.outfolder = outfolder
        self.eval_funcs = {"ks": self.ks, "dtfit": self.dtfit, "tfit_mit": self.tfit_mit, "meton":self.metonymy}
        if args_order is None: args_order = "head_verbs_args"
        self.args_order = args_order

    def metonymy(self):
        self.data = dutils.load_metonymy_dataset(self.datafile)
        items = []
        for s, v, o, e in self.data.keys():
            sbj = s[:-2]+"@N@SBJ"
            obj = o[:-2]+"@N@OBJ"
            items.append("{} {}".format(sbj, obj))
        # write generated file
        df = pd.DataFrame(data={"item": items, "target-relation": ["ROOT" for i in range(0, len(items))]})
        df.to_csv(os.path.join(self.outfolder, "{}.{}".format(self.name, self.args_order)), sep="\t", index=False)

    def ks(self):
        """
        It takes as input the original dataset and generates the file to be processed by SDM module.
        It also save a mapping file: the first id is the line of the original file, the other two
        correspond to the lines of the generated file.
        """
        self.data = pd.read_csv(self.datafile, delimiter="\t")

        events = []
        mapping = {}

        for ind in self.data.index:
            s1 = self.data["subject1"][ind]
            v1 = self.data["verb1"][ind]
            o1 = self.data["object1"][ind]

            s2 = self.data["subject2"][ind]
            v2 = self.data["verb2"][ind]
            o2 = self.data["object2"][ind]

            e1 = ""
            e2 = ""
            if self.args_order == "head_verbs_args":
                e1 = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(s1, v1, o1)
                e2 = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(s2, v2, o2)
            elif self.args_order == "verbs_args":
                e1 = "{}@V@ROOT {}@N@SBJ {}@N@OBJ".format(v1, s1, o1)
                e2 = "{}@V@ROOT {}@N@SBJ {}@N@OBJ".format(v2, s2, o2)
            else:
                sys.exit("Non acceptable order type")
            events.append(e1)
            events.append(e2)

            mapping[ind] = (len(events) - 2, len(events) - 1)

        # write generated file
        df = pd.DataFrame(data={"item": events, "target-relation": ["SENTENCE" for i in range(0, len(events))]})
        df.to_csv(os.path.join(self.outfolder, "{}.{}".format(self.name,self.args_order)), sep="\t", index=False)
        # write mapping file
        df = pd.DataFrame.from_dict(mapping, orient="index")
        df.to_csv(os.path.join(self.outfolder, "{}-mapping.csv".format(self.name.split(".")[0])), sep="\t",
                  header=None)

    def dtfit(self):
        """
        It takes as input the original self.dataset and generates the file to be processed by SDM module.
        """
        self.data = pd.read_csv(self.datafile, delimiter="\t")

        experiment = self.name.split(".")[0].split("_")[1]
        mapping = {"Triples": "OBJ", "Loc": "LOCATION", "Instr": "INSTRUMENT", "Time": "TIME", "Recipient": "RECIPIENT"}
        items = []
        for ind in self.data.index:
            s = self.data["SUBJECT"][ind]
            v = self.data["VERB"][ind]
            if self.args_order == "head_verbs_args":
                sv = "{}@N@SBJ {}@V@ROOT".format(s, v)
            elif self.args_order == "verbs_args":
                sv = "{}@V@ROOT {}@N@SBJ".format(v, s)
            else:
                sys.exit("Non acceptable order type")
            if experiment == "Triples":
                items.append(sv)
            else:
                items.append("{}  {}@N@OBJ".format(sv, self.data["OBJECT"][ind]))

        # write file
        df = pd.DataFrame(data={"item": items, "target-relation": [mapping[experiment] for i in range(0, len(items))]})
        df.to_csv(os.path.join(self.outfolder, "{}.{}".format(self.name, self.args_order)), sep="\t", index=False)

    def tfit_mit(self):
        self.data = pd.read_csv(self.datafile, delimiter="\t", header=None)

        items = []
        targets = []
        for ind in self.data.index:
            item, target = self.data[0][ind].split("_")
            item, pos = item.split("-")
            if pos == "v":
                items.append("{}@{}@ROOT".format(item, pos.upper()))
            else:
                items.append("{}@{}@SBJ".format(item, pos.upper()))
            targets.append((target.upper()))
        # write file
        df = pd.DataFrame(data={"item": items, "target-relation": targets})
        df.to_csv(os.path.join(self.outfolder, "{}.{}".format(self.name, self.args_order)), sep="\t", index=False)


def prepare_input_files(datafiles, outfolder, dataset_type, args_order):
    for datafile in datafiles:
        dataset = Datasets(datafile, outfolder, args_order)
        dataset.eval_funcs[dataset_type]()
