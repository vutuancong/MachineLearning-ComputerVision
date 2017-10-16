import tensorflow as tf 
import numpy as np 
from sklearn import datasets

digits = datasets.load_digits()

n_samples = len(digits.images)
image_data = digits.images.reshape((n_samples,-1))
feature_size = image_data.shape[1]
labels_size = 10

total_data_and_label = np.array(list(zip(image_data, digits.target)))

shuffle_indices = np.random.permutation(np.arange(n_samples))
shuffle_data = total_data_and_label[shuffle_indices]

data_size = int(0.9 * n_samples)

image_and_label = shuffle_data[:data_size]
test_data = shuffle_data[data_size:]

def convert_label(origin_labels):
	hot_labels = []
	for label_ in origin_labels:
		hot_label = [ 0 for _ in range(label_size)]
		hot_label[label_] = 1
		hot_labels.append(hot_label)
	return hot_labels

batch_size = 10
if data_size & batch_size == 0:
	num_batches_per_epoch = data_size / batch_size
else:
	num_batches_per_epoch = int(data_size / batch_size) +1

def batch_inter(_data, _batch_size, _batch_num):
	start_index = _batch_num * _batch_size
	end_index = min([(_batch_num + 1)* _batch_size, data_size])
	batch = _data[start_index:end_index]
	x_, y_ = zip(*batch)
	return list(x_), list(y_)