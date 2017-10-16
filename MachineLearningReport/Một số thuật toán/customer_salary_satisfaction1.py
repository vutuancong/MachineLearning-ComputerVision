import os
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from itertools import starmap

def load_data(file_path):
	data = []
	labels = []
	if os.path.isfile(file_path):
		f = open(file_path,'r')
		lines = f.readlines()
		print(lines)
		f.close()
		for line in lines:
			arr = line.split('\t')
			arr_float = starmap(float,arr)
			data.append(arr_float[:-1])
			labels.append(arr_float[-1])
	return data,labels

def train():
	x_data, y_data = load_data(file_path = 'customer_salary_satisfaction')
	x_train = x_data[:-1]
	y_train = y_data[:-1]
	lin_model = linear_model.linearRegression()
	lin_model.fit(x_train,y_train)
	return lin_model

def test():
	x_data, y_data = load_data(file_path= 'customer_salary_satisfaction')
	x_test = x_data[-1:]
	y_test = y_data[-1:]
	lin_model = train()
	y_predict = lin_model.predict(x_test)
	square_error = 0.5 * np.mean((y_predict - y_test) ** 2)
	print(square_error)
	plt.scatter(x_test, y_test, color = 'black' )
	plt.plot(x_test,y_predict,color = 'blue')
	plt.show()

if __name__ == '__main__':
	test()

