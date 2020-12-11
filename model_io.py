# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 12:44:34
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-04-22 12:06:25

import pandas as pd
import numpy as np
import pickle

class Model_pickle():
	def __init__(self):
		pass

	def dump(self, model, path, model_name):
		with open(path + '/' + model_name + '.pkl', 'wb') as f:
			pickle.dump(model, f)
		print('Done!')

	def load(self, path, model_name):
		with open(path + '/' + model_name + '.pkl', 'rb') as f:
			model = pickle.load(f)
		return model