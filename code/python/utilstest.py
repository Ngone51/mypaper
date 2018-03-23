# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from numpy import mat, array
from utils import topest, topk

class utilstest(unittest.TestCase):
	'''topest返回一个列表值最大的元素的下标'''
	def test_topest(self):
		a_list = [1, 4, 5, 3, 2]
		max_j = topest(a_list, [])
		self.assertEquals(max_j, 2)

	'''topk返回一个列表前k个值最大的元素的下标'''
	def test_topK_for_list(self):
		a_list = [1, 4, 5, 3, 2]
		topks = topk(a_list, 3)
		self.assertEquals(topks, [2, 1, 3])

	# '''topk返回一个numpy的array前k个值最大的元素的下标'''
	def test_topK_for_array(self):
		a = array([[1, 4, 5, 3, 2]])
		topks = topk(a[0].tolist(), 3)
		self.assertEquals(topks, [2, 1, 3])

	# '''topk返回一个numpy的matrix某一行的前k个值最大的元素的下标'''
	def test_topK_for_matrix(self):
		m = mat([[1, 4, 5, 3, 2], [6, 7, 9, 10, 8]])
		topks = topk(array(m)[0].tolist(), 3)
		self.assertEquals(topks, [2, 1, 3])	


# if __name__ == '__main__':
#     unittest.main()	




















	
