# -*- coding: utf-8 -*-  

#! /bin/python 

"""
  注意：由于我们之后又做了关注图G（N，E）有向边的实验，所以，数据文件的输入路径会在原先的
  输入路径前加“directed_”或“reverse_directed_”。注意区分数据的输入路径，不要搞混数据。
  该脚用于从data/(reverse_directed_/directed_)all_user_feature_vector.emd中抽取
  大V的特征向量。
  数据准备：
  1) data/(reverse_/directed_)all_user_feature_vector.emd：由node2vec根据所有用户的关注关系，
  生成的每个用户在该关注网络图的特征向量。注意，这里的所有用户包含在sub_big_v_graph
  中不存在的普通用户和大v用户。
  2) data/to_id_map.csv: 该文件存储了大v用户的id。
"""
import os
import sys
import numpy as np
from generate_sub_big_v_01_matrix import loadCSV

"""
  description: 从src_path存储的文件中抽取大V用户的特征向量存储到dst_path的文件中。
"""
def extract(src_path, dst_path):
	# In order to avoid our data file mistakenly deleted by this script,
	# we should do a fast-fail when we want to create a file which already exists.
	if os.path.exists(dst_path):
		raise Exception("%s alreay exists!" % dst_path)	
	# undirected
	# all_user_feature_vector_file_path = "../../data/all_user_feature_vector.emd"
	# directed(from_id -> to id)
	# all_user_feature_vector_file_path = "../../data/directed_all_user_feature_vector.emd"
	# directed(to_id -> from_id)
	# all_user_feature_vector_file_path = "../../data/reverse_directed_all_user_feature_vector.emd"

	# 加载to_id_map.csv
	to_id_map = loadCSV("../../data/to_id_map.csv")
	(m, cols_num) = to_id_map.shape

	# 用一个set()把所有大V的id储存起来，这样，我们就可以方便我们去判断
	# all_user_feature_vector.emd的一个特征向量是否是属于大V的。
	to_id_set = set()
	for i in xrange(m):
		to_id_set.add(to_id_map[i][1])

	# 打开all_user_feature_vector.emd
	f = open(src_path, "r")
	# 注意：输出路径是undirected还是directed还是reverse_directed
	# 创建big_v_feature_vector.emd, 用于存储大V的特征向量
	big_v_f = open(dst_path, "aw")


	firstline = True
	big_v_vectors = []
	for vector in f.readlines():
		# 第一行存储了该文件的meta信息，所以我们跳过它
		# see details：https://github.com/aditya-grover/node2vec
		if firstline:
			firstline = False
			pass
		id = int(vector.strip().split(" ")[0])
		# 如果该id正是大V的id，则我们把该特征向量提取出来
		if id in to_id_set:
			big_v_f.write(vector)

	f.close()
	big_v_f.close()

if __name__ == '__main__':
	src_path = sys.argv[1]
	dst_path = sys.argv[2]
	extract(src_path, dst_path)













