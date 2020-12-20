'''
Description: 
Version: 1.0
Autor: Troy Wu
Date: 2020-02-14 12:26:50
LastEditors: Troy Wu
LastEditTime: 2020-12-19 14:41:32
'''

import logging
import traceback

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

	def info(self, statement):
		logging.info(statement)
  
	def error(self):
		logging.error(traceback.format_exc())