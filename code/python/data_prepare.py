# -*- coding: utf-8 -*-  

#! /bin/python

import utils
from numpy import mat, array, delete
import random

"""
  @description: 加载大V的01关注矩阵(原始数据)，支持csv格式
  @parameter: delimiter, csv文件的分割符，默认为','
  @parameter: file_path, 原始数据的所在文件路径，默认'../../data/sub_big_v_01_matrix.csv'
  @return: 返回矩阵
"""
def loadX(delimiter = ",", file_path = "../../data/sub_big_v_01_matrix.csv"):
	# We pass 'False' to tell the function that we won't need the 0th col and 0th row
	# of the csv data.
	# And now, this matrix X do not have any redundant cols or rows.
	# 注意：在去除了冗余的第0行和第0列的数据之后的01关注矩阵（X）的第0行对应id为1的普通用户的关注信息，
	# 而第0列表示了id为1的大V用户的受关注情况。
	X = utils.loadCSV(file_path, delimiter, needZeroCol = False, needZeroRow = False)
	return array(X)

"""
  @description: 获取矩阵中所有的元素的值
  @parameter1: data, 要求数据类型为array
"""
def getAllSampleValue(data):
	sample_values = []
	(m, n) = data.shape
	for i in xrange(m):
		for j in xrange(n):
			sample_values.append(data[i, j])
	return sample_values		

"""
  @descrption: 获取所有的正样本(在矩阵中，值为'1'的坐标)列表  
  @parameter: data, 矩阵，原始数据集。
"""
def getPositiveSampleList(data):
	positive_sample_list = list()
	(m, n) = data.shape
	for i in xrange(m):
		for j in xrange(n):
			if data[i, j] == 1:
				positive_sample_list.append((i, j))
	return positive_sample_list

"""
  @descrption: 获取所有的正样本集合
  @parameter: data, 矩阵，原始数据集。
"""
def getPositiveSampleSet(data):
	return set(getPositiveSampleList(data))

"""
  @descrption: 如果row = True，则计算每行正样本的个数；如果row = False，则计算每列正样本的个。
  @parameter1: data, 矩阵，原始数据集
  @parameter2: row, True for row, False for col
"""
def countPositiveSample(data, row = True):
	(m, n) = data.shape
	l = m if row == True else n
	ones = [0 for x in xrange(l)]
	# 记录每一行/列正样本的个数
	for i in xrange(m):
		for j in xrange(n):
			if data[i, j] == 1:
				if row == True:
					ones[i] += 1
				else:
					ones[j] += 1	
	return ones

"""
  @description: 从所有正样本中，随机地抽取a * 100%的样本，作为移除信息
  @parameter1: positives, 所有正样本
  @parameter2: row_ones, 记录了原始数据集(矩阵)中每一行正样本的个数
  @parameter3: col_ones, 记录了原始数据集(矩阵)中每一列正样本的个数
  @parameter4: a(0 <= a <= 1), 信息移除的比例，默认0.4
"""
def getRemoveSet(positives, row_ones, col_ones, a = 0.4):
	# 准备去除的正样本的总数
	amount = int(len(positives) * a)
	remove_set = set()
	if amount > 0:
		# shuffle positive samples for random purpose
		random.shuffle(positives)
		for (p, q) in positives:
			# We need to ensure each row/col in matrix has at least one number of '1'
			# to avoid 'nan' error in gnmf algorithm.
			if row_ones[p] - 1 > 0 and col_ones[q] - 1 > 0:
				row_ones[p] -= 1
				col_ones[q] -= 1
				remove_set.add((p, q))
			if len(remove_set) == amount:	
				break
		if len(remove_set) < amount:
			print '''[warn]: Aim to remove %d positive samples totaly, but only %d can be removed at most due to "no zero line" limit.''' % (amount, len(remove_set))		
	return remove_set	

"""
  @description: 获取训练集
  @parameter1: subG, 包含普通用户和大V用户的关注图（原属数据集）。
  @parameter2: data_X, 01关注矩阵
  @parameter3: r_from_id_dict, 普通用户的current id到origianl id的映射表
  @parameter4: r_to_id_dict, 大V用户的current id到origianl id的映射表
  @parameter5: positives，原始数据集中的正样本集合（这里的正样本表示从普通用户到大V用户的直接关注关系）
  @parameter6: a(0 <= a <= 1), 从原始数据集中，去除原始信息的比例，默认0.4
"""
def getTrainX(subG, data_X, r_from_id_dict, r_to_id_dict, positives, a = 0.4):
	assert a >= 0 and a <= 1, "'a' must be 0 <= a <= 1"
	# 获取原始数据集每一行正样本的个数
	row_ones = countPositiveSample(data_X)
	# 获取原始数据集每一列正样本的个数
	col_ones = countPositiveSample(data_X, row = False)
	# 从所有正样本随机抽取部分作为需要移除的(关注)信息
	remove_set = getRemoveSet(positives, row_ones, col_ones, a)
	# first， remove the follow relations from data_X
	train_X = data_X.copy()
	for (p, q) in remove_set:
			train_X[p, q] = 0

	# then, we need to remove the same follow relations from subG 
	# to construct our TRUE train set
	train_set = subG.copy()
	remove_rows = []
	for (p, q) in remove_set:
		# note: though, we have a map between current ids and origianl ids already,
		# but there's still "1" gap difference betwen currnet id and data_X's row/col's 
		# index, as data_X's 0th row is correspond to current id "1", and same for col. 
		f_o_id = r_from_id_dict[p + 1]
		t_o_id = r_to_id_dict[q + 1]
		# we need to find the specific follow relation (f_o_id, t_o_id) in subG,
		# and this is rellay time consuming. We are eager to find a more efficient way.
		for row in xrange(subG.shape[0]):
			if int(subG[row][0]) == f_o_id and int(subG[row][1]) == t_o_id:
				remove_rows.append(row)
				break
	train_set = delete(train_set, remove_rows, axis = 0)			
	return (train_X, train_set)
























