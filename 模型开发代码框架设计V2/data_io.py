# -*- coding: utf-8 -*-
# @Author: Troy Wu
# @Date:   2020-02-10 21:31:07
# @Last Modified by:   Troy Wu
# @Last Modified time: 2020-02-17 11:36:18

import numpy as np
import pandas as pd
import teradatasql
import pymysql
from sqlalchemy import create_engine
import configparser

class Database():
	def __init__(self, path = 'config.ini'):
		#path是配置文件的地址
		config = congigparser.ConfigParser()
		config.read(path, encoding = 'utf-8')
		self.server = dict(config.items('server'))

	def read_data(self, sql, server = 'teradata'):
		if server == 'teradata':
			engine = create_engine('teradata://{}:{}@{}:{}/{}'.format(
				self.server['user'], self.server['password'], self.server['host'], self.server['port'], self.server['dbname']))

		if server = 'mysql':
			engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(
				self.server['user'], self.server['password'], self.server['host'], self.server['port'], self.server['dbname']))

		else:
			print('The server is wrong!')
			return None

		result = pd.read_sql(con = engine, sql = sql)
		print('Done!')
		return result

	def data_to_sql(self, data, server = 'teradata'):
		if server == 'teradata':
			engine = create_engine('teradata://{}:{}@{}:{}/'.format(
				self.server['user'], self.server['password'], self.server['host'], self.server['port']))

		if server == 'mysql':
			engine = create_engine('mysql+pymysql://{}:{}@{}:{}/'.format(
				self.server['user'], self.server['password'], self.server['host'], self.server['port']))

		else:
			print('The server is wrong!')
			return None

		data.to_sql(name = self.server['dbname'], con = engine, if_exits = 'append')
		print('Done!')