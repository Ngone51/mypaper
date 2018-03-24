# -*- coding: utf-8 -*-  

#! /bin/python 

from data_prepare import loadX, getPositiveSampleSet, getTrainX
from predict import train, topKbyRow, topKbyAll, popularity
from evaluation import f1, rmse, mae, accuracy, TPFPFN, precisionAndrecall
from numpy import array, zeros, dot, linalg, diag
from nmf import nmf
from gnmf import gnmf, computeW, computeD
from generate_big_v_feature_vector import extract
from utils import PAPER_DATA_PATH, NODE2VEC_ABSOLUTE_PATH, PAPER_PYTHON_PATH
import utils
import os

"""
  该脚本用于调参
"""


def adjust():
	subG = utils.loadCSV(file_path = "../../data/common_big_user_graph2.csv", \
		delimiter = " ", needZeroRow = True)
	# 由于subG和data_X中id不兼容，所以我们需要current id和original id的映射表
	# 分别加载from_id_map.csv和to_id_map.csv，并转换成dict
	(from_id, l) = utils.loadIdMapCsv("../../data/from_id_map.csv")
	(to_id, l) = utils.loadIdMapCsv("../../data/to_id_map.csv")
	r_from_id_dict = utils.transferIdMapToDict(from_id, reverse = True) 
	r_to_id_dict = utils.transferIdMapToDict(to_id, reverse = True)

	# 加载原始01关注矩阵（包含直接关注关系）
	print "Load data."
	data_X = loadX()
	(m, n) = data_X.shape
	total = m * n
	print "Load all positive samples in data."
	# 获取data_X的正样本集(这里的正样本表示从普通用户到大V用户直接的关注关系)
	positives = getPositiveSampleSet(data_X)
	
	# 组合参数
	# ks = [130, 135, 140, 145, 150, 155, 160, 165, 170]
	ks = [170]
	rs = [800]
	ss = [0.000035]
	outstanding_params = []
	params = []
	for i in xrange(len(ks)):
		for j in xrange(len(rs)):
			for t in xrange(len(ss)):
				params.append((ks[i], rs[j], ss[t]))

	tmp_files = []
	for i in xrange(len(params)):
		kk = params[i][0]
		ite = params[i][1]
		smooth = params[i][2]
		print "\n[ Exp%d: a = %f ]" % (i + 1, 0.4)
		# 获取训练集和train_X
		print "Generate training set."
		(train_X, train_set)= getTrainX(subG, data_X, r_from_id_dict, r_to_id_dict, list(positives), 0.4)
		
		# Now, we need to compute our W
		
		# we save train_set as csv data file in order to support node2vec's input request
		input_file = PAPER_DATA_PATH + "tmp_exp" + str(i + 1) + "_common_big_user_graph2.csv"
		output_file = PAPER_DATA_PATH + "tmp_exp" + str(i + 1) + \
		"_common_big_user_feature_vector_undirected2.emd"
		utils.saveCSV(input_file, train_set, delimiter = "	")
		tmp_files.append(input_file)
		# run node2vec
		cmd = "python " + NODE2VEC_ABSOLUTE_PATH + "src/main.py --input " + \
		input_file + " --output " + output_file + " > /dev/null 2>&1"
		print "run node2vec."
		os.system(cmd)
		tmp_files.append(output_file)
		# extract big V users's feature vectors from output_file 
		# by generate_big_v_feature_vector.py
		src_path = output_file
		dst_path = PAPER_DATA_PATH + "tmp_exp" + str(i + 1) + \
		"_big_v_feature_vector_undirected2.emd"
		extract(src_path, dst_path)
		tmp_files.append(dst_path)
		print "Compute W/D/L."
		# finally, we can compute W
		W = computeW(dst_path)
		D = computeD(W)
		L = D - W

		'''GNMF'''
		print "\n[ Methods: GNMF ]"
		# 训练模型
		(U, V) = gnmf(train_X, W, D, L, k = kk, r = ite, e = 0.1, s = smooth)
		res_X = dot(U, V.T)
		# 获取预测正样本集
		predict_positives = topKbyRow(res_X, 5)
		# predict_positives = topKbyAll(res_X, 655)
		(tp, fp, fn) = TPFPFN(positives, predict_positives)
		(p, r) = precisionAndrecall(tp, fp, fn)
		# f1 score 评估
		gf1 = f1(p, r)

		print "\n[ Methods: SVD ]"
		(U, s, V) = linalg.svd(train_X, full_matrices = False)
		S = diag(s)
		res_X = dot(dot(U, S), V)
		# 获取预测正样本集
		predict_positives = topKbyRow(array(res_X), 5)
		# predict_positives = topKbyAll(array(res_X), 655)
		(tp, fp, fn) = TPFPFN(positives, predict_positives)
		(p, r) = precisionAndrecall(tp, fp, fn)
		sf1 = f1(p, r)
	
		if gf1 >= sf1:
			outstanding_params.append(params[i])
			print "gf1: %f sf1: %f, k: %f, r: %f, s: %s" % (gf1, sf1, kk, ite, smooth)

		# 清理临时文件
		for tmp_file in tmp_files:
			os.remove(tmp_file)
		tmp_files = []
	outstanding_params_file = open(PAPER_DATA_PATH + "outstanding_params", "a+")	
	for param in outstanding_params:
		outstanding_params_file.write(str(param))
		outstanding_params_file.write("\n")

if __name__ == '__main__':
	adjust()









