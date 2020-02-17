# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-14 12:26:50
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-17 00:46:49

import logging

def save_log(logfile, statement)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	fh = logging.FileHandler(logfile, model = 'w')
	fh.setLevel(logging.INFO)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)

	logger.addHandler(fh)
	logger.addHandler(ch)

	logging.info(statement)