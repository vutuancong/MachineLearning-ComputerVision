from sklearn import datasets
import random
import numpy as np 
import tensorflow as tf 
import math

def load_train_data(split_precent = 0.9):
	#load data
	iris_data = datasets.load_iris()
	data_feature = iris_data.data
	data_labels = iris_data.target 
	
	#shuffle
	shuffle_indices = list(range(len(data_feature)))
	random.shuffle(shuffle_indices)
	data_feature_shuffle = np.array(data_feature)[shuffle_indices]
	data_labels_shuffle = np.array(data_labels)[shuffle_indices]

	#train data, load data

	split_indices = int(split_precent*len(data_feature_shuffle))

	feature_train = data_feature_shuffle[:split_indices]
	feature_test = data_feature_shuffle[split_indices:]

	labels_train = data_labels_shuffle[:split_indices]
	labels_test = data_labels_shuffle[split_indices:]

	return feature_train, labels_train, feature_test, labels_test

def convert_labels(labels):
	new_labels = []
	for label in labels:
		if label == 0:
			new_labels.append([1, 0, 0])
		elif label == 1:
			new_labels.append([0, 1, 0])
		else:
			new_labels.append([0, 0, 1])
	return new_labels


train_data, label_data_, test_data, test_label_ = load_train_data()
label_data = convert_labels(label_data_)
test_label = convert_labels(test_label_)

x = tf.placeholder('float', shape = [None, 4], name = 'input')
y_ = tf.placeholder('float', shape = [None, 3], name = 'label')

#hidden layer
Wh = tf.Variable(tf.zeros([4, 5]))
bh = tf.Variable(tf.zeros([5]))
h = tf.nn.sigmoid(tf.matmul(x, Wh) + bh)

#output layer(((5)))
W = tf.Variable(tf.zeros([5, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(h,W) + b)

init = tf.initialize_all_variables()

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for epoch in list(range(100)):
	for id_, ft in enumerate(train_data):
		feed_dict = {
			x: [ft],
			y_: [label_data[ id_]]

		}
		loss, _ = sess.run([cross_entropy, train_op], feed_dict = feed_dict)
		print('Step '+ str(id_) + ' loss: ' + str(loss))

predict_label = tf.argmax(y, 1)
true_label = tf.argmax(y_, 1)

correct_prediction = tf.equal(predict_label, true_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

result = sess.run([predict_label, true_label, correct_prediction], feed_dict = {x: test_data, y_: test_label})
accuracy_rs = sess.run(accuracy,feed_dict = {x: test_data, y_:test_label})

for i in list(range(len(result))):
	print('predict '+ str(result[i][0]) + ' real: '+ str(result[i][1]) + ' is correct: ' + str(result[i][2]))
print('accuracy '+ str(accuracy_rs))