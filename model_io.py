# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 12:44:34
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-17 00:46:08

import pandas as pd
import numpy as np
import pickle

class model_pickle():
	def __init__(self, path, model_name):
		self.path = path
		self.model_name = model_name

	def dump(self, model):
		with open(self.path + '/' + self.model_name + '.pkl', 'wb') as f:
			pickle.dump(model, f)
		print('Done!')

	def load(self):
		with open(self.path + '/' + self.model_name + '.pkl', 'rb') as f:
			model = pickle.load(f)
		return model