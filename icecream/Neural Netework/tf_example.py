# import tensorflow as tf 

# hello = tf.constant("This is constant")
# sess = tf.Session()

# print(sess.run(hello))

# a = tf.constant(1.0)
# b = tf.constant(2.0)

# s = a + b
# print(sess.run(s))

# import tensorflow as tf 
# from sklearn import datasets

# iris_data = datasets.load_iris()

# data = iris_data.data
# labels = iris_data.target

# x = tf.placeholder('float', shape =[1, 4], name = 'input')

# y = tf.multiply(x, 2, name = 'double')

# sess = tf.Session()

# for i in range(10):
# 	x_input = [data[i]]
# 	print('input: '+str(x_input))
# 	db = sess.run(y, feed_dict = {x: x_input})
# 	print('double: ' + str(db))

import tensorflow as tf

x = tf.Variable(tf.zeros([1, 4]))

inc_op = x.assign(tf.add(x, [1.0, 1.0, 1.0, 1.0]))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(10):
	x_test, _ = sess.run([x, inc_op])
	print(x_test)