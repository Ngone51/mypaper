# -*- coding: utf-8 -*-  

#! /bin/python 

from numpy import array, dot, shape, random, zeros
from math import pow

"""
  该脚本用于实现非负矩阵分解(NMF)算法
  分解的最终目标：X ≈ UV.T (V.T表示矩阵V的转置)
  目标函数：O = || X - UV.T ||^2
  迭代法则：u(i,k) = u(i,k) * (XV(i,k)) / (UV.TV(i,k))
          v(j,k) = v(j,k) * (X.TU(j,k)) / (VU.TU(j,k))
"""

"""
  @description 计算目标函数的当前值
"""
def lostfunction(X, U, V):
	t_X = dot(U, V.T)
	(m, n) = X.shape
	e = 0.0
	for i in xrange(m):
		for j in xrange(n):
			e += pow(X[i, j] - t_X[i, j], 2)
	return e


"""
  @description:  更新矩阵U
"""
def updateU(U, X, V):
	(m, k) = U.shape
	newU = zeros((m, k))
	nu = dot(X, V)
	de = dot(dot(U, V.T), V)
	for i in xrange(m):
		for j in xrange(k):
			newU[i, j] = U[i, j] * nu[i, j] / de[i, j]
			# TODO We have not found the root cause why nu[i, j]
			# will always result to be '0' in the iterations.
			# And, this is not a elegant way to solve this problem yet.
			# Maby, we can enrich our data set to introduce more 'follow relations',
			# especially for common user.

			# avoid 'nan' error
			if newU[i, j] == 0.0:
				newU[i, j] = 0.00000001
	return newU

"""
  @description: 更新矩阵V
"""
def updateV(V, X, U):
	(n, k) = V.shape
	newV = zeros((n, k))
	nu = dot(X.T, U)
	de = dot(dot(V, U.T), U)
	for i in xrange(n):
		for j in xrange(k):
			newV[i, j] = V[i, j] * nu[i, j] / de[i, j]
			if newV[i, j] == 0.0:
				newV[i, j] = 0.00000001
	return newV
			
"""
  @description: nmf算法
  @parameter1: X 初始矩阵
  @parameter2: k 矩阵分解后，两个子矩阵的列数和行数
  @parameter3: r 优化过程中，最大的迭代次数
  @parameter4: e 精度控制条件，当损失函数的值达到该精度时，
  				 可提前终止迭代
"""
def nmf(X, k = 100, r = 500, e = 0.00001):
	print "NMF start, k: %d, r: %d, e: %f" % (k, r, e)

	m, n = shape(X)

	# 先随机生成矩阵U和V
	U = array(random.random((m, k)))
	V = array(random.random((n, k)))

	print "Iteration start."
	for i in xrange(r):
		# 检查损失函数的是否达到要求的精度
		lost = lostfunction(X, U, V)
		if lost <= e:
			print '''We reach the target precision at iteration %d, and the
lostfunction() returned %f, which is less than target precision %f.''' % (i, lost, e)
			break
		U = updateU(U, X, V)
		# 实践证明，updateV中调用的'U'应该为迭代更新后的U
		V = updateV(V, X, U)
		if i > 0 and i % 100 == 0:
			print '''%dth iter ended, and current lost value is %f''' % (i, lost)
	print '''Iteration ended. Finally, the lost value is %f''' % lost
	print "NMF ended."

	return U, V


