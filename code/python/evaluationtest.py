# -*- coding: utf-8 -*-  

#! /bin/python

import unittest
from evaluation import precisionAndrecall

class evaluationtest(unittest.TestCase):
	
	''''''
	def test_precisionAndrecall(self):
		p = {(0, 0), (0, 1), (1, 0), (1, 1)}
		q = {(0, 1), (2, 2)}
		(precision, recall) = precisionAndrecall(p, q)
		self.assertEquals(precision, 0.5)
		self.assertEquals(recall, 0.25)








	
