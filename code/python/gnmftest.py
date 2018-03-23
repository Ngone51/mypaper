# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from numpy import array, mat, dot
from gnmf import computeD

class gnmftest(unittest.TestCase):
	'''用dot计算两个numpy的array相乘'''
	def test_array_dot(self):
		a = array([[1, 0, 0],
					[1, 1, 0]])
		b = array([[1, 0],
				   [0, 1],
				   [1, 0]])
		self.assertEquals(a.shape[1], b.shape[0])
		self.assertEquals((dot(a, b) == array([[1, 0], [1, 1]])).all(), True)

	'''用*计算两个numpy的array相乘, 抛出异常'''
	def test_array_multi(self):
		a = array([[1, 0, 0],
					[1, 1, 0]])
		b = array([[1, 0],
				   [0, 1],
				   [1, 0]])
		self.assertEquals(a.shape[1], b.shape[0])
		with self.assertRaises(ValueError):
			# ValueError: operands could not be broadcast together with shapes (2,3) (3,2)
			a * b

	'''用dot计算两个numpy的matrix相乘'''
	def test_array_dot(self):
		a = mat([[1, 0, 0],
					[1, 1, 0]])
		b = mat([[1, 0],
				   [0, 1],
				   [1, 0]])
		self.assertEquals(a.shape[1], b.shape[0])
		self.assertEquals((dot(a, b) == array([[1, 0], [1, 1]])).all(), True)		


	'''用*计算两个numpy的matrix相乘, 可行'''
	def test_array_dot(self):
		a = mat([[1, 0, 0],
					[1, 1, 0]])
		b = mat([[1, 0],
				   [0, 1],
				   [1, 0]])
		self.assertEquals(a.shape[1], b.shape[0])
		self.assertEquals(((a * b) == array([[1, 0], [1, 1]])).all(), True)


	''''''
	def test_computeD(self):
		w = array([[1, 2, 3],
				   [2, 4, 5],
				   [3, 5, 6]])
		d = computeD(w)
		self.assertEquals((d == array([[6, 0, 0],
								      [0, 11, 0],
								      [0, 0, 14]])).all(), True)










	
