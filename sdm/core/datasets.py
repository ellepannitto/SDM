import os
import pandas as pd


def ks(infile, datafolder):
	"""
	It takes as input the original dataset and generates the file to be processed by SDM module.
	It also save a mapping file: the first id is the line of the original file, the other two
	correspond to the lines of the generated file.

	:param infile: original dataset file
	:param datafolder: the dataset's folder (in /path/SDM/data/)

	"""
	data = pd.read_csv(infile, delimiter="\t")
	dataname = os.path.basename(infile).split(".")[0]
	outfolder = os.path.join(datafolder, "generated")

	events = []
	mapping = {}

	for ind in data.index:
		e1 = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(data["subject1"][ind], data["verb1"][ind], data["object1"][ind])
		e2 = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(data["subject2"][ind], data["verb2"][ind], data["object2"][ind])

		events.append(e1)
		events.append(e2)

		mapping[ind] = (len(events) - 2, len(events) - 1)

	# write generated file
	df = pd.DataFrame(data={"item": events, "target-relation": ["SENTENCE" for i in range(0, len(events))]})
	if not os.path.exists(outfolder): os.makedirs(outfolder)
	df.to_csv(os.path.join(outfolder, "{}.svo".format(dataname)), sep="\t", index=False)
	# write mapping file
	df = pd.DataFrame.from_dict(mapping, orient="index")
	df.to_csv(os.path.join(datafolder, "original/mapping.csv".format(dataname)), sep="\t", header=None)


def dtfit(infile, datafolder):
	"""
	It takes as input the original dataset and generates the file to be processed by SDM module.

	:param infile: original dataset file
	:param datafolder: the dataset's folder (in /path/SDM/data/)
	"""
	data = pd.read_csv(infile, delimiter="\t")
	outfolder = os.path.join(datafolder, "generated")

	experiment = os.path.basename(infile).split(".")[0].split("_")[1]
	mapping = {"Triples": "OBJ", "Loc": "LOCATION", "Instr": "INSTRUMENT", "Time": "TIME", "Recipient": "RECIPIENT"}
	items = []
	for ind in data.index:
		# organization@N@OBJ army@N@SBJ install@V@ROOT    OBJ
		sv = "{}@N@SBJ {}@V@ROOT".format(data["SUBJECT"][ind], data["VERB"][ind])

		if experiment == "Triples":
			items.append(sv)
		else:
			items.append("{}  {}@N@OBJ".format(sv, data["OBJECT"][ind]))

	# write file
	df = pd.DataFrame(data={"item": items, "target-relation": [mapping[experiment] for i in range(0, len(items))]})
	if not os.path.exists(outfolder): os.makedirs(outfolder)
	df.to_csv(os.path.join(outfolder, "{}.all".format(type)), sep="\t", index=False)


def nakov(infile, datafolder, order):
	"""
	It takes as input the original dataset and generates the file to be processed by SDM module.
	The parameter "order" specifies which are the items to be passed to the SDM.

	:param infile: original dataset file
	:param datafolder: the dataset's folder (in /path/SDM/data/)
	:param order:  [so, sv, vo, ov]
	"""
	data = pd.read_csv(infile, delimiter="\t", header=None)
	dataname = os.path.basename(infile)
	outfolder = os.path.join(datafolder, "generated")

	items = []

	if order == "so":
		compounds = set(data[0])
		for nn in compounds:
			nn = nn.split(" ")
			items.append("{}@N@SBJ {}@N@OBJ".format(nn[1], nn[0]))
		target = ["ROOT" for i in range(0, len(items))]

	elif order == "sv":
		for ind in data.index:
			compound = data[0][ind].split(" ")
			v = data[1][ind]
			items.append("{}@N@SBJ {}@V@ROOT".format(compound[1], v))
		target = ["OBJ" for i in range(0, len(items))]

	elif order == "vs":
		for ind in data.index:
			compound = data[0][ind].split(" ")
			v = data[1][ind]
			items.append("{}@V@ROOT {}@N@SBJ".format(v, compound[1]))
		target = ["OBJ" for i in range(0, len(items))]

	elif order == "vo":
		for ind in data.index:
			compound = data[0][ind].split(" ")
			v = data[1][ind]
			items.append("{}@V@ROOT {}@N@OBJ".format(v, compound[0]))
		target = ["SBJ" for i in range(0, len(items))]

	elif order == "ov":
		for ind in data.index:
			compound = data[0][ind].split(" ")
			v = data[1][ind]
			items.append("{}@N@OBJ {}@V@ROOT".format(compound[0], v))
		target = ["SBJ" for i in range(0, len(items))]

	# write file
	df = pd.DataFrame(data={"item": items, "target-relation": target})
	if not os.path.exists(outfolder): os.makedirs(outfolder)
	df.to_csv(os.path.join(outfolder, "{}.{}".format(dataname, order)), sep="\t", index=False)
