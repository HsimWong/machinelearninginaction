# coding: utf-8
'''
Created on Jan. 14, 2019
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

Note:
	Data files are compressed in a .zip file and is automatically
	extracted in the knn.py file
	
@author: Xin Wang, Beijing University of Technology
'''
import numpy as np 
from matplotlib import pyplot as plt
from os import listdir
import zipfile

# Dataset Creation
def createDataSet():
	# do not forget '[]' outside the data
	# The input variable of np.array should be a list of lists
	group = np.array([
		[1.0, 1.1], 
		[1.0, 1.0], 
		[0.0, 0.0], 
		[0.0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

''' 
 In the term of classification, there is no need for 
 nomalizing the result vector, since they are often not
 numerical.
'''
def autoNorm(dataSet):
	# normalizing the weight of each criteria influencing the training process
	# using 
	minimal = dataSet.min(0)
	maximum = dataSet.max(0)
	range_val = maximum - minimal
	normed_set = (dataSet - minimal) / range_val
	return normed_set, range_val, minimal


# dataset could be in any size of a matrix
# no critical line to be trained as we are 
# using the original data.
# This is a function containing both 
# training section as well as testing.
def classify0(inX, dataSet, labels, k):
	# Pick out the smallest k elements by scanning the data
	# Temporary elements are stored in a queue
	#	which is not properly implemented in Python
	def pick_min_k_ele(data, k):
		ret = []
		ret.append([data[0], 0])
		for i in range(len(data)):
			if not len(ret) == 0:
				if data[i] < ret[0][0]:
					if len(ret) == k:
						# Do not forget to input the position where 
						# elements to be popped.
						ret.pop(0)
					else:
						pass
					ret.append([data[i], i])
				else:
					pass
			else:
				pass
		return ret

	data_dim = dataSet.shape;

	# The ```tile()``` function is to shape the given matrix into 
	# certain dimension and size.
	# np.tile(Matrix, (int n, int m, int h, ...))
	# where second input variable represents 
	# times of duplication in coordinate direction
	data_diff = np.tile(inX, (data_dim[0], 1)) - dataSet
	# ```np.sum()``` is used to get the sum of values in certain dimension
	# np.sum(axis = int dim);
	# HOWEVER, **Transpose** in of 1-D matrix has no effect
	# Therefore, finally it gives a 1 * n matrix, which is a list
	dist = (data_diff ** 2).sum(axis = 1) ** 0.5

	# Simply sorting might not be a good way
	# scan the dataSet and simply pick out the smallest k elements
	k_near_list = pick_min_k_ele(dist, k)

	# Voting
	vote_count = {} 
	for dist_ind in k_near_list:
		vote = labels[dist_ind[1]]
		# dict.get(key, default_value) here returns " int 0" when the 
		# key does not exist, no need for key checks
		vote_count[vote] = vote_count.get(vote, 0) + 1

	# Methods given in the text book simply used Sort, which is O(nlogn)
	# in complexity. There is no need for such auxilury operation
	# However, I have not yet found proper way to scan a dict object
	# I simply cast the dict_key(list) into a list and go it through with
	# a complexity of O(n)
	
	key_list = list(vote_count.keys())
	max_val = vote_count[key_list[0]]
	max_label = key_list[0]
	for i in range(len(key_list) - 1):
		if vote_count[key_list[i + 1]] > max_val:
			max_val = vote_count[ key_list[ i + 1 ] ]
			max_label = key_list[i+1]
	return max_label

def file2matrix(filename):
	try:
		with open(filename) as f:
			raw_info = f.readlines()
	except:
		f = zipfile.ZipFile("./chap_01_data.zip", 'r')
		for file in f.namelist():
			f.extract(file,"./")
		with open(filename) as f:
			raw_info = f.readlines()
	labels = []
	dataSet = []
	for line in raw_info:
		parsed_line = line.strip().split('\t')
		dataSet.append(parsed_line[0:3])
		labels.append(parsed_line[3])
	labels = np.array(labels, dtype = 'str')
	dataSet = np.array(dataSet, dtype = 'float')
	return (dataSet, labels)

def PlotScatter():
	datingDataMat,datingLabels = file2matrix("datingTestSet.txt")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])
	plt.show()

def datingClassTest():
	train_mat, train_label = file2matrix('datingTestSet2.txt')
	normed_train_mat, normed_range, normed_min = autoNorm(train_mat)
	error_count = 0;
	total_test_count = len(normed_train_mat)

	# continue using the training set as the testing set
	# error rate is not necessarrily to be 0
	for k in range(100):
		error_count = 0
		for i in range(total_test_count):

			prd = classify0(normed_train_mat[i], normed_train_mat, train_label, k)
			
			if not prd == train_label[i]:
				error_count += 1
				# print('''Classifier came back with: %s, the real answer is: %s'''% (prd, train_label[i]))

		print("Total error rate is %f" % (error_count/float(total_test_count)))

def img2vector(filename):
	with open(filename) as f:
		returnVect = list(map(int, f.read().replace('\n', '')))
	
	label = filename[filename.find('Digits') + 7 : filename.find('_')]
	return ((returnVect), label)
 
def handwritingClassTest():
	# Load the training set
	try:
		train_name_list = listdir('trainingDigits')
	except:
		f = zipfile.ZipFile("./chap_01_data.zip", 'r')
		for file in f.namelist():
			f.extract(file,"./")
		train_name_list = listdir('trainingDigits')

	train_mat = []
	label_vec = []

	for i in range(len(train_name_list)):
		Mat, Lab = img2vector("./trainingDigits/" + train_name_list[i])
		label_vec.append(Lab)
		train_mat.append(Mat)
	train_mat = np.array(train_mat)

	# Load the testification set
	test_name_list = listdir('testDigits')
	total_num = len(test_name_list)
	for k in range(100):
		print("When k is chosen to be %s: " % (k + 1), end = "")
		error_count = 0
		for i in range(len(test_name_list)):
			vec, ans = img2vector("./testDigits/" + test_name_list[i])
			prd = classify0(np.array(vec), train_mat, label_vec, k + 1)

			if not ans == prd:

				error_count += 1

		print("Error rate: %s" % (error_count / float(total_num)))







if __name__ == '__main__':
	handwritingClassTest()