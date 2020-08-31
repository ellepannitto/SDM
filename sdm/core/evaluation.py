import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

import sdm.utils.data_utils as dutils

logger = logging.getLogger(__name__)


def sim_vecs(v1, v2):
	try:
		s = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))
	except ValueError:
		v1 = np.nan_to_num(v1)
		v2 = np.nan_to_num(v2)
		s = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))
	return s[0][0]


class Evaluation(object):
	def __init__(self, data_fpath, res_data_fpath, out_folder, vector_fpath=None, mapping_fpath=None):
		self.dataset = data_fpath
		self.res_data = res_data_fpath
		self.out_fold = out_folder
		self.mapping = mapping_fpath
		if vector_fpath:
			self.vecs = dutils.load_vectors(vector_fpath, len_vectors=-1)

		self.eval_funcs = {"ks": self.ks_evaluation, "dtfit": self.dtfit_evaluation, "tfit_mit": self.tfit_mit_evaluation}

	def ks_evaluation(self):
		# load mapping id
		mapping = pd.read_csv(self.mapping, delimiter="\t", header=None)
		map_id = {}
		for i in mapping.index:
			map_id[mapping[0][i]] = (mapping[1][i], mapping[2][i])
		# load dataset and output files
		data = pd.read_csv(self.dataset, delimiter="\t")

		res = pd.read_csv(self.res_data, delimiter="\t")
		sim_res = {}
		for k in sorted(map_id.keys()):
			v1 = np.array([float(x) for x in res["AC_vector"][map_id[k][0]].split()])
			v2 = np.array([float(x) for x in res["AC_vector"][map_id[k][1]].split()])

			sim = sim_vecs(v1, v2)
			sim_res[k] = sim

		# spermancorr
		scores = [data["score"][i] for i in sorted(sim_res.keys())]
		sim_scores = [sim_res[i] for i in sorted(sim_res.keys())]
		logger.info("Spearman's correlation: {}".format(spearmanr(scores, sim_scores)))
		print(spearmanr(scores, sim_scores))

		# write sims
		data["sims"] = sim_scores
		data.to_csv(os.path.join(self.out_fold, os.path.basename(self.dataset)+".sims"), index=False)

	def dtfit_evaluation(self):
		data = pd.read_csv(self.dataset, delimiter="\t")
		res = pd.read_csv(self.res_data, delimiter="\t")
		sim_res = {}
		for i in data.index:
			v_target = np.array([float(x) for x in res["AC_vector"][i].split()])
			try:
				label = res["target-relation"][i]
				if label == "OBJ":
					label = "OBJECT"
				v_original = self.vecs[data[label][i]]
				sim = sim_vecs(v_target, v_original)
				sim_res[i] = sim
			except KeyError:
				pass

		# spermancorr
		scores = [data["mean_rat"][i] for i in sorted(sim_res.keys())]
		sim_scores = [sim_res[i] for i in sorted(sim_res.keys())]
		logger.info("Spearman's correlation: {}".format(spearmanr(scores, sim_scores)[0]))

		# print(spearmanr(scores, sim_scores))

		# write sims
		data_out = data.ix[sorted(sim_res.keys())]
		data_out["sims"] = sim_scores
		out_fname = os.path.join(self.out_fold, os.path.basename(self.dataset) + ".sims")
		data_out.to_csv(out_fname, index=False)

	def tfit_mit_evaluation(self):
		# todo: complete
		print(1)


def compute_evaluation(original_data, results_data, out_dir, vector_file, mapping_file, data_type):
	e = Evaluation(original_data, results_data, out_dir, vector_file, mapping_file)

	e.eval_funcs[data_type]()
