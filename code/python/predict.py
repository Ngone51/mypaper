# -*- coding: utf-8 -*-  

#! /bin/python 

import utils
from utils import topest, topk, bottomk
from gnmf import gnmf
from numpy import mat, array, dot
from data_prepare import getPositiveSampleList, getAllSampleValue, countPositiveSample

"""
  @description: 加载大V的01关注矩阵
"""
def loadX(file_path = "../../data/sub_big_v_01_matrix_copy.csv"):
	# We pass 'False' to tell the function that we won't need the 0th col of the 
	# csv data, and the 0th row is default to be ignored(though, here, it's different 
	# compare to other csvs' ignorance reason.)
	# And now, this matrix X do not have any redundant cols or rows.
	X = utils.loadCSV(file_path, ",", False)
	return mat(X)


"""
  @description: 通过矩阵分解算法，训练数据集，并返回最后的结果集(分解后得到的两个矩阵的乘积)
  @parameter1: data_X, 原始数据集
  @parameter2: k, 矩阵分解后，两个子矩阵的列数和行数
  @parameter3: r, 优化过程中，最大的迭代次数
  @parameter4: e, 精度控制条件，当损失函数的值达到该精度时，
  				 可提前终止迭代
  @parameter5: s, smoothness(s >= 0，用于控制平滑度) 如果s为0，则gnmf等同于nmf	
"""

def train(train_X, k = 100, r = 500, e = 0.1, s = 0.1):
	# TODO 我们可以把训练方法通过参数的形式传经来，这样我们可以灵活地调用不同的训练方法
	# 训练模型
	(U, V) = gnmf(train_X, k, r, e, s)
	# 获取结果矩阵
	res_X = dot(U, V.T)
	return res_X

"""
  @description: 将原始数据集中，前k个正样本个数最多的列作为每一行上的预测正样本。
  				(在社交网络的推荐应用中，意味将关注度最高的前几个大V推荐给普通用户。)
  @parameter1: train_X, 训练集
"""
def popularity(train_X, k = 5):
	(m, n) = train_X.shape
	# 计算每一列正样本的个数
	ones = countPositiveSample(train_X, row = False)
	# 注意：ones的下标正好对应了数据集中的列的索引，所以我们才能调用topk
	assert len(ones) == n, "'ones' indexs do not be correspond with 'train_X''s cols."
	# 推荐正样本个数最多的前k个列
	predict_cols = topk(ones, k)
	predict_positives = set()
	for i in xrange(m):
		for j in  predict_cols:
			if (i, j) not in predict_positives:
				predict_positives.add((i, j))
	assert len(predict_positives) == k * m, \
		"expects %d predict positives, but got %d" %(k * m, len(predict_positives))

	return predict_positives

"""
  @description: 在结果集中，预测每行前k个值最大的样本为正样本，其余的均
  默认预测为负样本
  @prameter: res_X, 结果集
  @parameter: k
  @return：预测为正样本的样本集(k * m个样本的坐标集合，其中m为res_X的行数)
"""
def topKbyRow(res_X, k):
	(m, n) = res_X.shape
	assert k < n and k >= 1, \
		"k needs to be equal or greater than 1 and less than n."
	predict_positives = set()
	# get a copy of 'res_X' to avoid polluting the 'res_X'
	X = res_X.copy()
	for i in xrange(m):
		kk = k
		indexs = topk(X[i].tolist(), kk)
		for j in indexs:
			if (i, j) not in predict_positives:
				predict_positives.add((i, j))
	assert k * m == len(predict_positives), \
		"expects %d predict positives, but got %d" %(k * m, len(predict_positives))
	return predict_positives

"""
  @description: 在结果集中，选择所有的样本值的第k个大小为阈值，超过该阈值的样本预测为正样本，其余的均
  预测为负样本
  @prameter: res_X, 结果集
  @parameter: k, k is very important.
  @return：预测为正样本的样本集
"""
def topKbyAll(res_X, k):
	values = getAllSampleValue(res_X)
	# 对所有value从大到小排序
	values.sort(reverse = True)
	# get our threshold
	threshold = values[k - 1]
	
	predict_positives = set()
	(m, n) = res_X.shape
	for i in xrange(m):
		for j in xrange(n):
			if res_X[i, j] >= threshold and (i, j) not in predict_positives:
				predict_positives.add((i, j))
	return predict_positives			

"""
  @desciption: 根据给定的预测方法(topKbyRow or topKbyAll)，返回预测地正样本集
  @parameter1: f, 预测方法
  @parameter2: res_X, 结果集
  @parameter3: k, 当f为topKbyRow时，表示要选取前k个为正样本；当f为topKbyAll时，k为所有样本的第k个值。
"""
def getPredictPositives(f, res_X, k):
	predict_positives = f(res_X, k)
	return predict_positives


def bottomKbyRow(res_X, k):
	(m, n) = res_X.shape
	assert k < n and k >= 1, \
		"k needs to be equal or greater than 1 and less than n."
	predict_positives = set()
	# get a copy of 'res_X' to avoid polluting the 'res_X'
	X = res_X.copy()
	for i in xrange(m):
		kk = k
		indexs = bottomk(X[i].tolist(), kk)
		for j in indexs:
			if (i, j) not in predict_positives:
				predict_positives.add((i, j))
	assert k * m == len(predict_positives), \
		"expects %d predict positives, but got %d" %(k * m, len(predict_positives))
	return predict_positives	
