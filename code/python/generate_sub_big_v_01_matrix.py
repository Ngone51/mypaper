# -*- coding: utf-8 -*-  

#! /bin/python 

"""
  该脚用于生成大v和普通用户之间的01关注矩阵。
  数据准备：
  1) data/sub_big_v_graph.csv: from_id -> to_id, from_id为普通用户id，而
  to_id为大V id.
  2) from_id_map.csv: from_id到1-n(n为普通用户总数)的映射
  3) to_id_map.csv: to_id到1-m(m为大V用户总数)的映射
"""

import numpy as np

"""
  @description: 加载csv文件，并通过numpy转换成数组
  @parameter：file_path: csv文件的路径
"""
def loadCSV(file_path):
	tmp = np.loadtxt(file_path, dtype = np.str, delimiter = ",")
	# 1:，表示从csv文件的第二行开始取数据，因为第一行是每列的列名
	# 0:，表示从csv文件的第一列开始取数据
	data_list = tmp[1:, 0:].astype(np.int)
	return data_list

"""
  @description: 通过numpy将一个数据转换成csv格式的文件，并存储到指定路径
  @parameter：file_path: csv文件的存储路径
  @parameter：data: 将要转换的数据
"""
def saveCSV(file_path, data):
	# fmt用于控制数据的输出格式，类似于c语言里的prinf的格式控制
	# 可以是‘%10.5f’、‘%d’等等
	np.savetxt(file_path, data, fmt = "%d", delimiter = ",")	

if __name__ == '__main__':
	# 加载sub_big_v_graph.csv
	sub_big_v_graph = loadCSV("../../data/sub_big_v_graph.csv")
	# 加载from_id_map.csv
	from_id_map = loadCSV("../../data/from_id_map.csv")
	# 加载to_id_map.csv
	to_id_map = loadCSV("../../data/to_id_map.csv")

	# edges表示在sub_big_v_graph.csv中，表示从普通用户到大V用户关注关系的个数
	(edges, cols_num) = sub_big_v_graph.shape

	# n is the number of common user, and m is the number of big v
	(n, cols_num) = from_id_map.shape
	(m, cols_num) = to_id_map.shape

	# 我们需要把从数据库中export出来的，csv格式的id映射关系，转换成
	# 在python真正的映射结构——dict()
	from_id_dict = dict()
	for i in xrange(n):
		from_id_dict[from_id_map[i][1]] = from_id_map[i][0]

	to_id_dict = dict()
	for i in xrange(m):
		to_id_dict[to_id_map[i][1]] = to_id_map[i][0]



	# Now, we can create an 2 dim array as 0-1 follow martrix, though,
	# it is all zero yet.
	# we init array as m +1 and n +1, because ids from from_id_map and
	# to_id_map start from '1'. So, the array's 0th col and 0th row default
	# to be zero, and we shall not access thoes areas.
	sub_big_v_follow_list = [[0 for x in xrange(m + 1)] for y in xrange(n + 1)]

	for edge in xrange(edges):
		# get an follow relation edge from sub_big_v_graph,
		# then, get from_id and to_id separately.
		from_id = sub_big_v_graph[edge][0]
		to_id = sub_big_v_graph[edge][1]
		# transfer original id to 1-n/m id
		current_from_id = from_id_dict[from_id]
		current_to_id = to_id_dict[to_id]
		# set big_v_follow_array's correspond element to '1', since
		# it represent an follow relation between common user and big v.
		sub_big_v_follow_list[current_from_id][current_to_id] = 1

	saveCSV("../../data/sub_big_v_01_matrix.csv", sub_big_v_follow_list)

