import numpy as np 
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import  LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import os

__global_map = map
lmap = lambda func, *iterable: list(__global_map(func,*iterable)) 
map = lmap

__global_range = range
lrange = lambda stop: list(__global_range(stop))
range = lrange

def load_data(file_path):
	if(os.path.isfile(file_path)):
		f = open(file_path,"r")
		lines = f.readlines()
		f.close()
		feature_data = []
		labels_data = []
		for line in lines:
			str_arr = line.split('\t')
			float_arr =  map(float,str_arr)
			feature_data.append(float_arr[:-1])
			labels_data.append(float_arr[-1:])
#		print(labels_data)
	else:
		print('File is not exist')

	return feature_data, labels_data

#load_data("data.txt")

def load_train_data(file_path):
	data, labels = load_data(file_path)
	shuffle_indices = range(len(data))
	# random.shuffle(shuffle_indices)
	shuffle_data = np.array(data)[shuffle_indices]
	shuffle_labels = np.array(labels)[shuffle_indices]
	split_index = int(0.8 * len(data))
	data_train = shuffle_data[:split_index]
	label_train = shuffle_labels[:split_index]
	data_test = shuffle_data[split_index:]
	label_test = shuffle_labels[split_index:]
	return data_train, label_train, data_test, label_test

def train(file_path):
	# gnb = GaussianNB()
	# gnb  = LogisticRegression()
	# gnb = KNeighborsClassifier()
	gnb = QuadraticDiscriminantAnalysis()
	
	train_data, train_labels, _, _ = load_train_data(file_path)
	# print(len(train_data)),
	train_labels = np.ravel(train_labels)
	model = gnb.fit(train_data, train_labels)
	# fw =  open("model.pkl","wb")
	# pickle.dump(model, fw)
	# fw.close()
	# print(model)
	return model

# train('data.txt')

def test(file_path):
	_, _, test_data, test_labels = load_train_data(file_path)
	# fr = open("model.pkl","rb")
	# model = pickle.load(fr)
	model = train(file_path)
	# fr.close()
	prefict_labels = model.predict(test_data)
	total = 0
	count = 0
	for id_,label_ in enumerate(prefict_labels):
		if label_ == test_labels[id_]:
			count +=1
		total +=1
	accuracy = float(count) / total
	print(accuracy)

def main():
	train('data.txt')
	test('data.txt')


if __name__ == '__main__':
	main()

map = __global_map
range = __global_range