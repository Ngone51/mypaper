# -*- coding: utf-8 -*-  

#! /bin/python 

"""
  该脚本用于生成普通用户和大V用户的关注图subG(n, e)
  注意：当前，我们选择的普通用户和大V用户仍然是sub_big_v_01_matrix.csv里的用户，
  或者分别和from_id_map.csv和to_id_map.csv里的用户对应。即普通用户193个，
  大V用户244个，从普通用户到大V用户的直接关注关系655条。
  这是我们的数据准备过程的历史遗留问题。这样做可能会带来的坏处是，数据量过小，推荐效果不理想。
  我们先把新的计算W的方式给写出来，然后再看看推荐效果。数据有不合适的地方，我们后期再改进。
"""

import utils

def main():
	# 加载在关注图subG中,所有普通用户(from_id)和大V用户(to_id)的id
	(from_id, l) = utils.loadIdMapCsv("../../data/from_id_map.csv")
	(to_id, l) = utils.loadIdMapCsv("../../data/to_id_map.csv")
	from_id_dict = utils.transferIdMapToDict(from_id) 
	to_id_dict = utils.transferIdMapToDict(to_id)

	from_original_id_set = set(from_id_dict.keys())
	to_original_id_set = set(to_id_dict.keys())

	# 加载follow_graph.csv(由于数据历史遗留问题，我们当前只能这样处理更方便点),
	follows = utils.loadCSV("../../data/follow_graph.csv", delimiter = "	", needZeroRow = True)
	# 在common_big_user_follows中不仅包含了普通用户到大V用户的关注关系，还包含了普通用户之间，大V用户
	# 之间，以及大V用户到普通用户的关注关系。
	common_big_user_follows = []
	index = 0
	for i in xrange(len(follows)):
		f_id = follows[i][0]
		t_id = follows[i][1]
		# 如果这条关注关系中的两个用户都在普通用户或大V用户中出现过，则将其添加到我们的subG中
		if (f_id in from_original_id_set or f_id in to_original_id_set) and \
		(t_id in from_original_id_set or t_id in to_original_id_set):
			common_big_user_follows.append([])
			# 为了能通过numpy.savetxt来存储csv格式的文件，我们要以二维list的形式存储数据
			common_big_user_follows[index].append(f_id)
			common_big_user_follows[index].append(t_id)
			index += 1
	utils.saveCSV("../../data/common_big_user_graph2.csv", common_big_user_follows, delimiter = " ")
	print "success !"

if __name__ == '__main__':
	main()
