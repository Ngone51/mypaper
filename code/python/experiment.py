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

def main():
	# 实际证明，如果我们把关注图G(N,E)当作无向图，交由node2vec处理，提取大V用户的特征向量效果最好。
    # 如果把普通用户到大V用户的关注方向作为关注图G的有向的方向，那么最后的效果极差。如果把前面的方向
    # 反过来，则效果也很差，还比不上传统的NMF。
	print "Start experiment(undirected)."
	# 效果极差
	# print "Start experiment(directed: from_id -> to_id)."
	# 效果很差
	# print "Start experiment(reverse directed: to_id -> from_id)."

	# 注意：理论上，我们需要在加载完subG之后，再从subG删除一定比例的关注信息，然后再分别构建
	# 01关注矩阵（即data_X）和计算相似度矩阵W。但是，由于历史遗留原因（以及可能地，“zero line”
	# 的限制），我们提前构建好了01关注矩阵，不需要再通过subG去构建。所以，现在我们要做的是：从01
	# 关注矩阵随机删除一定比例的关注信息，而这部分信息也同样需要在subG中删除。然后，我们再去用删除
	# 部分信息后的subG去计算相似度矩阵W。
	# 另外需要注意的是，一些变量或名词含义的变更：
	# subG: 普通用户和大V用户之间的关注图，包含了所有关注关系。这是我们的原始数据集。其中，用户的id
	#       仍为original id
	# data_X: 普通用户和大V用户之间的01关注矩阵，包含了直接的关注关系。这不再是原始数据集。
	#		  其中，矩阵的第0行表示current id为“1”的普通用户的关注情况。而矩阵的第1行表示
	#        current id为“1”的大V用户的受关注情况。
	# train_X: train_X是data_X删除部分关注信息后的“训练集”。现在，它并不是真正意义上的训练集。真正的
	#          训练集应该是subG删除部分关注信息后的数据集。但是为了保留代码的兼容性，我们不对现在存有
	#          歧义的变量名做修改。我们需要自己明确其中的差别，必要之处也要作出说明。
	# train_set: 是subG删除部分直接的关注关系后的数据集，是真正的训练集。事实上，train_set包含了train_X  
	# 正样本: 之前我们说的正样本（也就是data_X中值为“1”的样本)其实是subG中属于直接关注关系的样本。
	# 
	# subG和data_X的id不统一，是因为subG还需要通过node2vec计算，然后再通过
	# generate_big_v_feature_vector.py抽取大V用户的特征这个流程所限制的。由此造成了不兼容。

	'''data prepare'''
	# 加载关注图subG(n, e)（原始数据集common_big_user_graph2.csv）
	# subG应该是np.array类型
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
	# W = computeW(big_v_feature_vector_file_path = "../../data/reverse_directed_big_v_feature_vector.emd")
	# W = computeW(big_v_feature_vector_file_path = "../../data/directed_big_v_feature_vector.emd")
	# W = computeW(big_v_feature_vector_file_path = "../../data/big_v_feature_vector.emd")
	# W = computeW(big_v_feature_vector_file_path = "../../data/big_v_feature_vector_undirected2.emd")
	# D = computeD(W)
	# L = D - W

	# 根据不同的关注关系的去除比例，比较不同方法的推荐效果
	# 实际上，我们的数据集最多只能删除0.58的关注关系（由于“zero line”的限制）
	percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
	# percentages = [0.4]
	tmp_files = []
	for i in xrange(len(percentages)):
		print "\n[ Exp%d: a = %f ]" % (i + 1, percentages[i])
		# 获取训练集和train_X
		print "Generate training set."
		(train_X, train_set)= getTrainX(subG, data_X, r_from_id_dict, r_to_id_dict, list(positives), percentages[i])
		
		# Now, we need to compute our W
		
		# we save train_set as csv data file in order to support node2vec's input request
		input_file = PAPER_DATA_PATH + "tmp_exp" + str(i + 1) + "_common_big_user_graph2.csv"
		output_file = PAPER_DATA_PATH + "tmp_exp" + str(i + 1) + \
		"_common_big_user_feature_vector_undirected2.emd"
		utils.saveCSV(input_file, train_set, delimiter = " ")
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
		(U, V) = gnmf(train_X, W, D, L, k = 130, r = 800, e = 0.1, s = 0.004)
		res_X = dot(U, V.T)
		# 获取预测正样本集
		predict_positives = topKbyRow(res_X, 5)
		# predict_positives = topKbyAll(res_X, 655)
		(tp, fp, fn) = TPFPFN(positives, predict_positives)
		(p, r) = precisionAndrecall(tp, fp, fn)
		# f1 score 评估
		f1(p, r)
		rmse(fp, fn, total)
		mae(fp, fn, total)

		# # '''NMF'''
		# print "\n[ Methods: NMF ]"
		# # TODO 把nmf封装到predict.py中去
		# (U, V) = nmf(train_X, r = 500)
		# res_X = dot(U, V.T)
		# # 获取预测正样本集
		# predict_positives = topKbyRow(res_X, 5)
		# # predict_positives = topKbyAll(res_X, 655)
		# (tp, fp, fn) = TPFPFN(positives, predict_positives)
		# (p, r) = precisionAndrecall(tp, fp, fn)	
		# f1(p, r)
		# rmse(fp, fn, total)
		# mae(fp, fn, total)

		print "\n[ Methods: SVD ]"
		(U, s, V) = linalg.svd(train_X, full_matrices = False)
		S = diag(s)
		res_X = dot(dot(U, S), V)
		# 获取预测正样本集
		predict_positives = topKbyRow(array(res_X), 5)
		# predict_positives = topKbyAll(array(res_X), 655)
		(tp, fp, fn) = TPFPFN(positives, predict_positives)
		(p, r) = precisionAndrecall(tp, fp, fn)
		f1(p, r)
		rmse(fp, fn, total)
		mae(fp, fn, total)

		# print "\n[ Methods: popularity ]"
		# '''popularity'''
		# predict_positives = popularity(train_X, 5)
		# (tp, fp, fn) = TPFPFN(positives, predict_positives)
		# (p, r) = precisionAndrecall(tp, fp, fn)	
		# f1(p, r)
		# rmse(fp, fn, total)
		# mae(fp, fn, total)

	# 清理临时文件
	for tmp_file in tmp_files:
		os.remove(tmp_file)
if __name__ == '__main__':
	main()