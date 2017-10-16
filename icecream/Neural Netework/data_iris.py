# import data_helpers

from sknn.mlp import Layer, Classifier
from sklearn import datasets
import random
import numpy as np

# __global_map = map 
# lmap = lambda func,*iterable:list(__global_map(func,iterable))
# map = lmap

def load_train_data(split_percent = 0.9):
	data_iris = datasets.load_iris()
	feature_data = data_iris.data
	labels_data = data_iris.target
	
	shuffle_indices = list(range(len(feature_data)))
	random.shuffle(shuffle_indices)
	shuffle_feature_data = feature_data[shuffle_indices]
	shuffle_labels_data = labels_data[shuffle_indices]

	split_indices = int(split_percent*(len(shuffle_feature_data)))
	train_data = (shuffle_feature_data)[:split_indices]
	test_data = (shuffle_feature_data)[split_indices:]
	train_labels = (shuffle_labels_data)[:split_indices]
	test_labels = (shuffle_labels_data)[split_indices:]

	return train_data, train_labels, test_data, test_labels


def convert_labels(labels):
	new_labels = []
	for label in labels:
		if label == 0:
			new_labels.append([1, 0 ,0])
		elif label == 1:
			new_labels.append([0, 1, 0])
		else:
			new_labels.append([0, 0, 1])

	return new_labels

def train(train_data, train_labels):
	#First layer
	hidden_layer = Layer(type = 'Sigmoid', name = 'hidden', units = 10)
	
	#Second layer
	out_layer = Layer(type = 'Softmax', name = 'output')
	layers = [hidden_layer, out_layer]
	mlp = Classifier(layers = layers, random_state = 1)
	mlp.fit(train_data, train_labels)

	print(mlp)
	
	return mlp

def test(test_data, test_labels, model):
	predict_labels = model.predict(test_data)
	total = 0
	count = 0
	for id_, label in enumerate(predict_labels):
		if max(label) == max(test_labels[id_]):
			count+=1
		total+=1
	print('accuracy '+ str(float(count) / total))

def main():
	feature_train, label_train, feature_test, label_test = load_train_data()
	label_train = convert_labels(label_train)
	label_test = convert_labels(label_test)
	feature_train = np.array(feature_train)
	label_train = np.array(label_train)
	feature_test = np.array(feature_test)
	label_test =np.array(label_test)
	mlp = train(feature_train, label_train)
	test(feature_test, label_test, mlp)


if __name__ == '__main__':
	main()
