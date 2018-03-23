# -*- coding: utf-8 -*-  

#! /bin/python 

import utils
import random
from numpy import power
from math import pow

"""
  @description: 计算准确率
"""
def accuracy(FP, FN, total):
	print "accuracy: %f" % ((total - FP - FN) / total)

"""
  @description: 计算Root Mean Suqare Error(均方根误差)
  @parameter1: FP
  @parameter2: FN
  @parameter3: 数据集的样本总数
  note: 由于在我们的实验中，正负样本的值均为0和1，所以我们的均方误差可以直接用FP+FN来计算
"""
def rmse(FP, FN, total):
	print "rmse: %f" % pow((FP + FN) / total, 0.5)

"""
  @descritpion: 计算Mean Absolute Error(平均绝对误差)
  @parameter1: FP
  @parameter2: FN
  @parameter3: 数据集的样本总数
  note: 由于在我们的实验中，正负样本的值均为0和1，所以我们的绝对误差可以直接用FP+FN来计算
"""
def mae(FP, FN, total):
	print "mae: %f" % ((FP + FN) / total)

"""
  @description: 计算TP、FP、FN
  @parameter1: positives, 原始数据集中的正样本集
  @parameter2: predict_positives, 预测为正样本的样本集
  note: TP: 把正样本预测为正样本的个数； FP: 把负样本预测为正样本的个数；
        FN: 把正样本预测为负样本的个数
"""
def TPFPFN(positives, predict_positives):
	TP = FP = FN = 0.0
	for (i, j) in predict_positives:
		# 预测为正样本的样本在原始数据集中确实为正样本
		if (i, j) in positives:
			TP += 1.0
		else:  # 预测为正样本的样本在原始数据集为负样本
			FP += 1.0

	for (i, j) in positives:
		# 原始数据集中的正样本被预测为负类
		if (i, j) not in predict_positives:
			FN += 1.0

	print "TP: %f, FP: %f, FN: %f" % (TP, FP, FN)		
	return (TP, FP, FN)		

"""
  @description: 根据TP、FP、FN计算精确率和召回率
"""
def precisionAndrecall(TP, FP, FN):
	precision = TP / (TP + FP)		
	recall = TP / (TP + FN)
	print "precision: %f, recall: %f" % (precision, recall)
	return (precision, recall)

"""
  @description: 计算f1 score
  @parameter1: precision, 精确率
  @parameter2: recall, 召回率
"""
def f1(precision, recall):
	f1_score = (precision * recall) / (precision + recall) * 2
	print "f1: %f" % f1_score
	return f1_score












