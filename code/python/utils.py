# -*- coding: utf-8 -*-  

#! /bin/python 

import numpy as np

"""
  @description: 通过numpy将一个数据转换成csv格式的文件，并存储到指定路径
  @parameter1：file_path: csv文件的存储路径
  @parameter2：data: 将要转换的数据. 1D or 2D array_like.
  @parameter3：delimiter, 数据存储的分隔符
"""
def saveCSV(file_path, data, fmt = "%d", delimiter = ","):
	# fmt用于控制数据的输出格式，类似于c语言里的prinf的格式控制
	# 可以是‘%10.5f’、‘%d’等等
	np.savetxt(file_path, data, fmt = fmt, delimiter = delimiter)

"""
  @description: 加载csv文件，并通过numpy转换成数组
  @parameter1：file_path: csv文件的路径
  @parameter2: delimiter: csv文件的数据分隔符
  @parameter3: needZeroCol: 是否需要第0列。对于sub_big_v_01_matrix.csv，我们不需要第0列，
  				而对于xxx_id_map.csv，我们需要第0列。
  @parameter4: needZeroRow: 是否需要第0行。默认为False。对于follow_graph.csv我们需要第0行。				
"""
def loadCSV(file_path, delimiter = ",", needZeroCol = True, needZeroRow = False):
	tmp = np.loadtxt(file_path, dtype = np.str, delimiter = delimiter)
	startCol = 0 if needZeroCol else 1
	startRow = 0 if needZeroRow else 1

	# if we can controll the astype() it would be better, then we will not
	# to transfer type in other place, e.g. in transferIdMapToDict()
	data_list = tmp[startRow:, startCol:].astype(np.float)
	return data_list

"""
  加载to_id_map.csv
"""
def loadIdMapCsv(file_path):
	try:
		id_map = np.array(loadCSV(file_path))
		(n, cols_num) = id_map.shape
		assert cols_num == 2, "The number of id_map's col must be 2."
		return (id_map, n)
	except IOError:
		raise IOError("Can not load file %s !" % file_path)

"""
  @description: 转换id_map为python中的dict
  @parameter1: id_map, np.array类型
  @paramtere2: reverse, reverse为False, 则id_dict存储original_id到current_id的映射；
               若reverse为True，则id_dict存储current_id到original_id的映射。
  Note: id_map的列数必须为两列，且左边一列为current_id, 右边一列为original_id
"""
def transferIdMapToDict(id_map, reverse = False):
	# n为id_map的行数，cols_num为列数。
	(n, cols_num) = id_map.shape
	assert cols_num == 2, "The number of id_map's col must be 2."
	id_dict = dict()
	for i in xrange(n):
		if reverse:
			id_dict[int(id_map[i][0])] = int(id_map[i][1])
		else:	
			id_dict[int(id_map[i][1])] = int(id_map[i][0])
	return id_dict


"""
  @description: 在可遍历的values对象中，查找值最大的元素，并返回对应下标
  @parameter1: values, 可遍历的数据类型
  @return 返回最大值对应的下标
  note: 'values' will be changed here, so the caller should pass a values's copy
  if it does not meant to change 'values'. This except in tests. 
"""
def topest(values, exists):
	max_setted = False
	max_v = -1.0
	max_i = -1
	for i in xrange(len(values)):
		if max_setted == False or (values[i] > max_v and max_setted == True):
			if i not in exists:
				max_v = values[i]
				max_i = i
				max_setted = True
	if max_setted == True:
		return max_i
	else:
		raise Exception("Do not find a topest value.")

"""
  @description: 在values中，寻找前k个值最大的元素，并返回对应的下标。
  @parameter1: values, 要求数据类型为'list'
  @parameter2: k
  @return 返回前k个最大值对应的下标的列表

"""
def topk(values, k):
	# Yet, we need to ensure that the type of 'values' must be 'list'.
	assert isinstance(values, (list)), "'values' needs to be 'list'."
	topks = []
	# note: we need to get a deep copy from 'values' to avoid
	# change the original data. 
	values_copy = values[:]
	while k > 0 :
		i = topest(values_copy, set(topks))
		topks.append(i)
		k -= 1
	return topks


def bottomest(values, exists):
	min_setted = False
	min_v = 99999
	min_i = -1
	for i in xrange(len(values)):
		if min_setted == False or (values[i] < min_v and min_setted == True):
			if i not in exists:
				min_v = values[i]
				min_i = i
				min_setted = True
	if min_setted == True:
		return min_i
	else:
		raise Exception("Do not find a topest value.")

def bottomk(values, k):
	# Yet, we need to ensure that the type of 'values' must be 'list'.
	assert isinstance(values, (list)), "'values' needs to be 'list'."
	bottomks = []
	# note: we need to get a deep copy from 'values' to avoid
	# change the original data. 
	values_copy = values[:]
	while k > 0 :
		i = bottomest(values_copy, set(bottomks))
		bottomks.append(i)
		k -= 1
	return bottomks

PAPER_ABSOLUTE_PATH = "/Users/wuyi/workspace/paper/"	
PAPER_DATA_PATH = PAPER_ABSOLUTE_PATH + "data/"
PAPER_PYTHON_PATH = PAPER_ABSOLUTE_PATH + "code/python/"

NODE2VEC_ABSOLUTE_PATH = "/Users/wuyi/soft/node2vec/"

