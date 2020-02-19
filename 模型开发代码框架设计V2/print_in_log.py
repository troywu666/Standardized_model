# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-14 12:26:50
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-19 12:12:45

import logging

class Save_log():
	def __init__(self, path, logfile_name):
		logger = logging.getLogger()
		logger.setLevel(logging.INFO)

		logfile = path + '/' + logfile_name + '.log'
		fh = logging.FileHandler(logfile, mode = 'a')
		fh.setLevel(logging.INFO)

		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)

		formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)

		logger.addHandler(fh)
		logger.addHandler(ch)

	def logging(self, statement):
		logging.info(statement)