# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from numpy import array, dot
from nmf import nmf

class gnmftest(unittest.TestCase):
	def test_nmf_1(self):
		a = array([[0.1, 0.2, 0.3],
				   [0.4, 0.5, 0.6],
				   [0.7, 0.8, 0.9]])
		(U, V) = nmf(a, k = 2)
		print "U:", U
		print "V:", V
		print "U * V:", dot(U, V.T)

	def test_nmf_2(self):
		a = array([[1, 2, 3],
				   [4, 5, 6],
				   [7, 8, 9]])
		(U, V) = nmf(a, k = 2)
		print "U:", U
		print "V:", V
		print "U * V:", dot(U, V.T)	