import tensorflow as tf
import numpy as np 
from sklearn import datasets

digits = datasets.load_digits()

n_samples =  len(digits.images)
image_size = digits.images.shape[1]

image_data = digits.images.reshape(n_samples, -1)
feature_size = image_data.shape[1]
label_size = 10

total_data_and_label = np.array(list(zip(image_data, digits.target)))

shuffle_indices = np.random.permutation(np.arange(n_samples))
shuffle_data = total_data_and_label[shuffle_indices]

data_size = int(0.9*n_samples)

image_and_label = shuffle_data[:data_size]
test_data = shuffle_data[data_size:]

def convert_label(origin_labels):
	hot_labels = []
	for label_ in origin_labels:
		hot_label = [0 for _ in range(label_size)]
		hot_label[label_] = 1
		hot_labels.append(hot_label)
	return hot_labels

batch_size = 10
if data_size % batch_size == 0:
	num_batches_per_epoch = data_size / batch_size
else:
	num_batches_per_epoch = int(data_size / batch_size) + 1

def batch_iter(_data, _batch_size, _batch_num):
	start_index = _batch_num * _batch_size
	end_index = min([(_batch_num + 1) *_batch_size, data_size])
	batch = _data[start_index:end_index]
	x_, y_ = zip(*batch)
	return list(x_), list(y_)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1 , shape = shape)
	return tf.Variable(initial)

def conv2d(_x, _W):
	return tf.nn.conv2d(_x,_W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(_x):
	return tf.nn.max_pool(_x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

x = tf.placeholder('float',shape = [None, feature_size], name = 'input')
x_image = tf.reshape(x, [-1, image_size, image_size ,1])

y_ = tf.placeholder('float',shape = [None, label_size], name = 'label')
# First  conv

Wh1 = weight_variable([3, 3, 1, 32])
bh1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, Wh1) + bh1)
h_pool1 = maxpool2d(h_conv1)

#Second conv
Wh2 = weight_variable([2, 2, 32, 64])
bh2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, Wh2) + bh2)
h_pool2 = maxpool2d(h_conv2)
full_feature_size = image_size // 4
h_pool_flat = tf.reshape(h_pool2, [-1, full_feature_size * full_feature_size*64])

Wfull1 = weight_variable([full_feature_size * full_feature_size * 64, 1024])
bfull1 = bias_variable([1024])

h_full = tf.nn.relu(tf.matmul(h_pool_flat, Wfull1) + bfull1)

Wfull2 = weight_variable([1024, label_size])
bfull2 = bias_variable([label_size])

y = tf.nn.softmax(tf.matmul(h_full, Wfull2) + bfull2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#test label
true_label = tf.argmax(y_, 1)
predict_label = tf.argmax(y, 1)
correct_prediction = tf.equal(true_label, predict_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

for epoch in range(100):
	if epoch %	10 == 0:
		x_batch, y_batch = zip(*test_data)
		x_batch = list(x_batch)
		y_batch = convert_label(list(y_batch))
		acc = sess.run(accuracy, feed_dict = {x: x_batch, y_: y_batch})
		print('accuracy: ' + str(acc))

	for i in range(num_batches_per_epoch):
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffle_data = image_and_label[shuffle_indices]
		batch_load = batch_iter(shuffle_data, batch_size, i)
		x_batch = batch_load[0]
		y_batch = convert_label(batch_load[1])
		sess.run(train_op, feed_dict = {x: x_batch, y_: y_batch})
	print('--epoch--: '+ str(epoch))