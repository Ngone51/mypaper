# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from numpy import mat, array
from predict import topKbyRow, topKbyAll, popularity

class predicttest(unittest.TestCase):
	'''topKbyRow返回每一行前k个最大值的下标的集合'''
	def test_topKbyRow(self):
		a = array([[1, 2, 0],
				   [3, 1, 4],
				   [1, 5, 6]])
		indexs = topKbyRow(a, 2)
		self.assertEquals(len(indexs), 6)
		self.assertEquals(indexs, {(0, 1), 
								  (0, 0),
								  (1, 2),
								  (1, 0),
								  (2, 2),
								  (2, 1)})

	'''推荐最受欢迎(关注度最高)的前k个大V用户给普通用户'''
	def test_popularity(self):
		a = array([[1, 0, 0],
					  [1, 1, 0],
					  [1, 1, 1]])
		# 根据推荐的结果，预测的正样本集
		predict_positives = popularity(a, 2)
		self.assertEquals(len(predict_positives), 3 * 2)
		self.assertEquals(predict_positives, {(0, 0),
											  (1, 0),
											  (2, 0),
											  (0, 1),
											  (1, 1),
											  (2, 1)})
	def test_topKbyAll(self):
		a = array([[1, 2, 0],
				   [3, 1, 4],
				   [1, 5, 6]])
		indexs = topKbyAll(a, 4)
		self.assertEquals(len(indexs), 4)
		self.assertEquals(indexs, {(1, 0), 
								  (1, 2),
								  (2, 1),
								  (2, 2)})







































	
