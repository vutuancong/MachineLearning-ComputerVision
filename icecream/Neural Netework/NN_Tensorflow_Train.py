from sklearn import datasets
import numpy as np 
import tensorflow as tf 

def train_test_data(split_percent = 0.9):
	iris_data = datasets.load_iris():
	data_feature = iris_data.data 
	data_labels = iris_data.target

	shuffle_indices = list(range(len(data_feature)))
	random.shuffle(shuffle_indices)

	shuffle_feature_data = data_feature[shuffle_indices]
	shuffle_lables_data = data_labels[shuffle_indices]

	#train vs test

	split_indices = int(split_percent*len(data_feature))

	train_data = np.array(shuffle_feature_data)[:split_indices]
	test_data = np.array(shuffle_feature_data)[split_indices:]

	train_labels = np.array(shuffle_lables_data)[:split_indices]
	test_labels = np.array(shuffle_lables_data)[split_indices:]

	return train_data, train_labels, test_data, test_labels

def  cover_labels(labels):
	new_labels = []
	for label in labels:
		if(label == 0):
			new_labels.append([1,0,0])
		if(label == 1):
			new_labels.append([0,1,0])
		if(label == 2):
			new_labels.append([0,0,1])

	return new_labels

train_data, train_label, test_data, test_label = train_test_data()

label_data = convert_labels(train_label)
label_train = convert_labels(test_label)

x = tf.placeholder('float', shape = [None, 4], name = 'input')
y_ = tf.placeholder('float', shape = [None, 3], name = 'label')

# tao them 1 layer
# Wh = tf.Variable(tf.zeros([4,5]))
# bh = tf.Variable(tf.zeros([5]))
# h = tf.nn.sigmoid(tf.matmul(x, Wh) + bh)

W = tf.Variable(tf.zeros[4,3])
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

init = tf.initialize_all_variables()

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

trian_op = tf.train.GadientDescentOptimizer(0.01).mininize(cross_entropy)

sess = tf.Session()
sess.run(init)

for epoch in list(range(100)):
	for id_, ft in enumerate(train_data):
		feed_dict= {
		x:[ft],
		y_: [label_data][id_]
		}
		loss, _ = sess.run([cross_entropy, train_op], feed_dict = feed_dict )
		print('Step' + str(id_) + 'logss' + str(loss))
predict_label = tf.argmax(y, 1)
true_label  = tf.argmax(y_,1)

#chuyen dang correct_prediction
correct_prediction = tf.equal(predict_label, true_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

result = sess.run([predict_label, true_label, correct_prediction], feed_dict = {x: test_data, y_:test_label})
accuracy_rs = sess.run(accuracy,feed_dict = {x:test_data, y_:test_label})

for i in list(range(len(result))):
	print('predict' + str(result[i][0]) + 'real: '+ str(result[i][1] + 'is correct: '+ str(result[i][2])))
print('accuracy: '+ str(accuracy_rs))