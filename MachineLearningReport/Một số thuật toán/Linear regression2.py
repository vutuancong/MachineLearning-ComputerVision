import os
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

__global_map = map #keep reference to the original map
lmap = lambda func, *iterable: list(__global_map(func, *iterable)) #using map here will cause infinite recursion
map = lmap

def load_data(file_path ='customer_salary_satisfaction'):
	data = []
	labels = []
	if os.path.isfile(file_path):
		f = open(file_path,'r')
		lines = f.readlines()
		f.close()
		for line in lines:
			arr = line.split('\t')
			arr_float = map(float,arr)#test
			data.append(arr_float[:-1])
			labels.append(arr_float[-1:])
			# print(line)
	else:
		print('File is not exist')
	return data,labels
# map = __global_map #restore the original map
# load_data('customer_salary_satisfaction')

def train():
	x_data, y_data = load_data(file_path = 'customer_salary_satisfaction')
	# print(x_data)
	x_train = x_data[:-200]
	y_train = y_data[:-200]
	lin_model = linear_model.LinearRegression()
	# print(lin_model)
	lin_model.fit(x_train,y_train)
	return lin_model

# # load_data('customer_salary_satisfaction')

def test():
	x_data, y_data = load_data(file_path = 'customer_salary_satisfaction')
	x_test = x_data[-200:]
	y_test = y_data[-200:]
	lin_model = train()
	y_predict = lin_model.predict(x_test)
	square_error = 0.5 * np.mean ((y_predict - y_test)**2)
	print(square_error)
	plt.scatter(x_test,y_test,color = 'black')
	plt.plot(x_test,y_predict,color = 'blue')
	plt.show()

if (__name__ == '__main__'):
	test()