# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from numpy import mat, array
from data_prepare import countPositiveSample, getPositiveSampleList, getAllSampleValue, getTrainX

class data_preparetest(unittest.TestCase):
	'''当row = True，计算每行正样本的个数'''
	def test_countPositiveSample_by_row(self):
		matrix = mat([[1, 0, 0],
					  [1, 1, 0],
					  [0, 0, 1]])
		ones = countPositiveSample(matrix, row = True)
		self.assertEquals(ones[0], 1)
		self.assertEquals(ones[1], 2)
		self.assertEquals(ones[2], 1)

	'''当row = False，计算每列正样本的个数'''
	def test_countPositiveSample_by_col(self):
		matrix = mat([[1, 0, 0],
					  [1, 1, 0],
					  [0, 0, 1]])
		ones = countPositiveSample(matrix, row = False)
		self.assertEquals(ones[0], 2)
		self.assertEquals(ones[1], 1)
		self.assertEquals(ones[2], 1)


	def test_getTrainX(self):
		a = array([[1, 0, 1],
				   [1, 1, 0],
				   [0, 0, 1]])
		positives = getPositiveSampleList(a)
		(m, n) = a.shape
		train_X = getTrainX(a, positives, 0.8)
		train_positives = getPositiveSampleList(train_X)
		self.assertEquals(len(train_positives), 4)

	def test_getAllSampleValue(self):
		a = array([[0.1, 0.2, 0.3],
			       [0.4, 0.5, 0.6]])
		values = getAllSampleValue(a)
		self.assertEquals(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])	       	

# if __name__ == '__main__':
#     unittest.main()	




















	
