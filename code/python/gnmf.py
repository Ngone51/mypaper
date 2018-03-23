# -*- coding: utf-8 -*-  

#! /bin/python 

from numpy import mat, array, zeros, ones, random, dot, trace, shape, power
from math import pow 
from math import e as mathe
import predict
import utils

"""
  该脚本用于实现Graph Regularized NMF(GNMF)算法
  分解的最终目标：X ≈ UV.T (V.T表示矩阵V的转置)
  目标函数：O = || X - UV.T ||^2 + λTr(V.TLV) (Tr()用于求一个矩阵的迹，
  L = D - W，D(j,j) = ∑(j)W(j,l), W为矩阵X中每个列之间的相似度矩阵)
  迭代法则：u(i,k) = u(i,k) * (XV(i,k)) / (UV.TV(i,k))
          v(j,k) = v(j,k) * ((X.TU + λWV)(j,k)) / ((VU.TU + λDV)(j,k))
"""

"""
  @description: 计算(big_v之间的)相似度矩阵
  @paramteter1: big_v_feature_vector_file_path，大V用户特征向量的存储路径
  @paramteter2: pi，计算热核函数时，pi的值
"""
def computeW(big_v_feature_vector_file_path = "../../data/big_v_feature_vector.emd", pi = 2):
	# 准备好big v的id映射字典。因为，在big v的特征向量中，存储的是original id，
	# 现在，我们需要将其准换为current id。
	# n为big_v的个数
	(big_v_id_map, n) = utils.loadIdMapCsv("../../data/to_id_map.csv")
	big_v_id_dict = utils.transferIdMapToDict(big_v_id_map)

	# 加载big v的特征向量
	# n+1: because our ids start from '1'.
	big_v_feature_vector_list = [0] * (n + 1)
	try:
		big_v_feature_vector_f = open(big_v_feature_vector_file_path, "r")
		for vector in big_v_feature_vector_f.readlines():
			big_v_original_id = int(vector.strip().split(" ")[0])
			vector = array(vector.strip().split(" ")[1:], dtype = float)
			big_v_feature_vector_list[big_v_id_dict[big_v_original_id]] = vector
	except IOError:
		raise IOError("Can not open file %s !" % big_v_feature_vector_file_path)
	finally:
		big_v_feature_vector_f.close()
	

	# 创建大小为(n+1, n+1)的相似度矩阵W, 且初始化每个元素为0
	W = zeros((n + 1, n + 1))
	for i in xrange(1, n + 1):
		for j in xrange(1, n + 1):
			# W是一个对称矩阵，所以我们只需要算一半就可以了
			if i <= j:
				# note：必须保证W[i,j] > 0, 以避免在迭代过程中，发生U或V小于0的情况
				# 为了能使矩阵的分解达到收敛，在这里，我们用热核函数来计算两个向量的相似度。
				# 之前使用，欧式距离(可能需要用点积)计算两个向量之间的相似度，并不能使迭代过程收敛。
				W[i, j] = pow(mathe, power(big_v_feature_vector_list[i] - big_v_feature_vector_list[j], 2).sum() / -pi)
				# Oops! I forget to set W[j, i], and this leads to trace(V.T * L * V) less than 0.
				W[j, i] = W[i, j]

	# TODO 我们可以尝试只选择每个大V的p个相似度最高的大V之间的W[i,j], 
	# 而把和其它大V之间的W[i,j]置为0。看看这样的效果是否会变得更好。
	# neighbors = predict.bottomKbyRow(W, p + 1)
	# for i in xrange(n):
		# for j in xrange(n):
			# if (i, j) not in neighbors:
				# W[i, j] = 0.0

	# we should also remove the redundant rows and cols in W						
	return array(W[1:, 1:])

"""
  @description: 计算矩阵D。矩阵D是一个对角矩阵，对角矩阵上的每个元素是矩阵W的每个列或行的和。
"""
def computeD(W):
	(n, n) = W.shape
	D = zeros((n, n))
	for i in xrange(n):
		D[i, i] = W[i].sum()
	return D				
	
"""
  @description 计算目标函数的当前值
"""
def lostfunction(X, U, V, L, s):
	tX = dot(U, V.T)
	(m, n) = X.shape
	e1 = 0.0
	e1 = power(tX - X, 2).sum()
	# Note: we need	to compute the trace first, don't forget about it.
	e2 = s * (trace(dot(dot(V.T, L), V)))
	return e1 + e2


"""
  @description 更新矩阵U。
"""
def updateU(U, X, V):
	(m, k) = shape(U)
	floor = 0.000001
	nu = dot(X, V)
	de = dot(dot(U, V.T), V)
	for i in xrange(m):
		for j in xrange(k):
			U[i, j] = U[i, j] * nu[i, j] / max(de[i, j], floor)
	return U	
			
"""
  @description 更新矩阵V。
"""
def updateV(V, X, U, W, D, s):
	(n, k) = shape(V)
	floor = 0.000001
	nu = dot(X.T, U) + s * dot(W, V)
	de = dot(dot(V, U.T), U) + s * dot(D, V)
	for i in xrange(n):
		for j in xrange(k):
			V[i, j] = V[i, j] * nu[i, j] / max(de[i, j], floor)
	return V		
			
"""
  @description: gnmf算法
  @parameter1: X, 初始矩阵
  @parameter2: W, 相似度矩阵。在我们的实验中，表示大V用户之间的相似度矩阵。
  @parameter3: D
  @parameter4: L, L = D - W
  @parameter5: k, 矩阵分解后，两个子矩阵的列数和行数
  @parameter6: r, 优化过程中，最大的迭代次数
  @parameter7: e, 精度控制条件，当损失函数的值达到该精度时，
  				 可提前终止迭代
  @parameter8: s, smoothness(s >= 0，用于控制平滑度) 如果s为0，则gnmf等同于nmf	 
"""
def gnmf(X, W, D, L, k, r ,e = 0.1, s = 0.1):
	print "GNMF start, k: %d, r: %d, e: %f, s: %f" % (k, r, e, s)

	# 先随机生成矩阵U和V
	(m, n) = X.shape
	U = array(random.random((m, k)))
	V = array(random.random((n, k)))


	print "Iteration start."
	for i in xrange(r):
		# 检查损失函数的是否达到要求的精度
		lost = lostfunction(X, U, V, L, s)
		
		if lost <= e:
			print '''We reach the target precision at iteration %d, and the
lostfunction() returned %f, which is less than target precision %f.''' % (i, lost, e)
			break

		# 更新U，V
		U = updateU(U, X, V)
		V = updateV(V, X, U, W, D, s)

		if i > 0 and i % 100 == 0:
			print '''%dth iter ended, and current lost value is %f''' % (i, lost)
	print '''Iteration ended. Finally, the lost value is %f''' % lost
	print "GNMF ended."
	return U, V
		

if __name__ == '__main__':
	# only for test

	# We pass 'False' to tell the function that we won't need the 0th col of the 
	# csv data, and the 0th row is default to be ignored(though, here, it's different 
	# compare to other csvs' ignorance reason.)
	# And now, this matrix X do not have any redundant cols or rows.
	X = utils.loadCSV("../../data/sub_big_v_01_matrix_copy.csv", ",", False)
	print "X.shape: ", X.shape
	print "X:"
	print X
	(U, V, lost) = gnmf(X, 100, 100, 1, 0.1)
	print '''gnmf() ended. Finally, the lost value is %f''' % lost
	print "U * V.T:", dot(U, V.T)
	