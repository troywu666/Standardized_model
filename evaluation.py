'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-11 19:48:02
LastEditors: Troy Wu
LastEditTime: 2020-12-29 10:42:42
'''
# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-11 19:48:02
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-12-01 17:20:18

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, recall_score, precision_score, roc_curve
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import seaborn as sn
from scikitplot.metrics import plot_confusion_matrix, plot_calibration_curve, plot_roc, plot_ks_statistic, plot_precision_recall, plot_lift_curve, plot_cumulative_gain
from functools import reduce

class Metrics():
	def __init__(self, y_true, y_pred):
		self.y_true = y_true

		if isinstance(y_pred, dict):
			self.y_pred1 = True
			self.y_pred2 = True
		else:
			if y_pred.ndim == 1:
				self.y_pred2 = np.array([1- y_pred, y_pred]).T
				self.y_pred1 = y_pred
			if y_pred.ndim == 2:
				self.y_pred2 = y_pred
				self.y_pred1 = y_pred[:, 1]

	def eval_score(self):
		return {'accuracy_score': accuracy_score(self.y_true, self.y_pred1.round()), \
				'recall_score': recall_score(self.y_true, self.y_pred1.round()), \
				'precision_score': precision_score(self.y_true, self.y_pred1.round()), \
				'roc_auc_score': roc_auc_score(self.y_true, self.y_pred1), \
				'f1_score': f1_score(self.y_true, self.y_pred1.round())}

	def eval_plot(self):
		f, axes = plt.subplots(nrows = 1, ncols = 7, figsize = (64, 9))
		plot_roc(self.y_true, self.y_pred2, ax = axes[0])
		plot_confusion_matrix(self.y_true, self.y_pred1.round(), ax = axes[1])
		plot_ks_statistic(self.y_true, self.y_pred2, ax = axes[2])
		plot_precision_recall(self.y_true, self.y_pred2, ax = axes[3])
		plot_lift_curve(self.y_true, self.y_pred2, ax = axes[4])
		plot_cumulative_gain(self.y_true, self.y_pred2, ax = axes[5])
		plot_calibration_curve(self.y_true, [self.y_pred2], ax = axes[6])
		#plt.show()
	
	def get_best_threshold(self):
		fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred1)
		index = np.argmin((fpr**2 + (tpr-1)**2))
		return thresholds[index]


class Metrics_comparison(Metrics):
	def __init__(self, y_true, y_pred):
		Metrics.__init__(self, y_true, y_pred)
		self.items = y_pred.items()

	def compare_score(self):
		compare_frame = pd.DataFrame()
		for item, pred in self.items:
			if pred.ndim == 1:
				self.y_pred2 = np.array([1 - pred, pred]).T
				self.y_pred1 = pred
			if pred.ndim == 2:
				self.y_pred2 = pred
				self.y_pred1 = pred.argmax(axis = 1)
			compare_frame[item] = self.eval_score().values()
		compare_frame.index = ['accuracy_score', 'recall_score', 'precision_score', 'roc_auc_score', 'f1_score']
		#compare_frame.plot(kind = 'bar')
		return compare_frame

	def compare_plot(self):
		for item, pred in self.items:
			if pred.ndim == 1:
				self.y_pred2 = np.array([1 - pred, pred]).T
				self.y_pred1 = pred
			if pred.ndim == 2:
				self.y_pred2 = pred
				self.y_pred1 = pred.argmax(axis = 1)
			self.eval_plot()
			#plt.show()

def weightedKS(y_true, y_pred, weight = 1):
	if weight == 1:
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
	elif isinstance(weight, list):
		lis = [i for i in zip(y_pred, y_true, weight)]
		reversed_lis = sorted(lis, keys = lambda x: [0], reverse = True)
		pos = np.cumsum(np.array([w for (p, y, w) in reversed_lis if y > 0.5]))
		neg = np.cumsum(np.array([w for (p, y, w) in reversed_lis if y < 0.5]))
		pos_all = reduce(lambda x, y: x+y, [w for (p, y, w) in reversed_lis if y > 0.5])
		neg_all = reduce(lambda x, y: x+y, [w for (p, y, w) in reversed_lis if y <= 0.5])
		tpr = pos / pos_all
		fpr = neg / neg_all
	return max(tpr-fpr)