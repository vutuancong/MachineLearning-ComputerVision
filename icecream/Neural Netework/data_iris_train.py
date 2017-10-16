from sknn.mlp import Layer, Classifier
from sklearn import datasets
import numpy as np 
import random

def load_train_data(split = 0.9):
	iris_data = datasets.load_iris()

	feature_data = iris_data.data
	labels_data = iris_data.target

	shuffle_indices = list(range(len(feature_data)))
	random.shuffle(shuffle_indices)

	shuffle_feature_data  = feature_data[shuffle_indices]
	shuffle_labels_data = labels_data[shuffle_indices]

	split_indices = int(split*len(shuffle_feature_data))
	train_data = shuffle_feature_data[:split_indices]
	test_data = shuffle_feature_data[split_indices:]
	train_label = shuffle_labels_data[:split_indices]
	test_label = shuffle_labels_data[split_indices:]

	return train_data, train_label, test_data, test_label

def covert_label(labels):
	new_label = []
	for label in labels:
		if label == 0:
			new_label.append([1, 0 ,0])
		elif label == 1:
			new_label.append([0, 1, 0])
		else:
			new_label.append([0, 0, 1])
	return new_label

def train_data(train_data, train_label):
	#layer 1
	hidden_layer = Layer(type = 'Sigmoid', name = 'Hidden', units = 10)
	#layer 2
	out_layer = Layer(type = 'Softmax', name = 'out')
	layer = [hidden_layer, out_layer]
	#train
	mlp = Classifier(layers = layer, random_state = 1)
	mlp.fit(train_data, train_label)
	
	print (mlp)

	return mlp

def test(test_label, test_data, model):
	predict_labels = model.predict(test_data)
	total = 0
	count = 0
	for id_, label in enumerate(predict_labels):
		if max(label) == max(test_label[id_]):
			count+=1
		total+=1
	print("accuracy " + str(float(count/total)))

def main():
	train_data, train_label, test_data, test_label = load_train_data()
	train_label = covert_label(train_label)
	test_label = covert_label(test_label)
	train_data = np.array(train_data)
	train_label = np.array(train_label)
	test_data = np.array(test_data)
	test_label = np.array(test_label)
	mlp = train_data(train_data, train_label)
	test(test_label, test_data, mlp)

if __name__ == '__main__':
	main()
