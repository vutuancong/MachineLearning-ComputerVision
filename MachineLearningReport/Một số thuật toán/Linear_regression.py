import numpy as np 
import os
from sklearn import linear_model
import matplotlib.pyplot as plt 

__global_map = map 
lmap = lambda func, *iterable: list(__global_map(func,*iterable))
map = lmap

def load_data(file_path):
	if(os.path.isfile(file_path)):
		f = open(file_path,'r')
		lines = f.readlines()
		f.close()
		feature_data = []
		labels_data = []
		for line in lines:
			str_arr = line.split('\t')
			float_arr = map(float,str_arr)
			feature_data.append(float_arr[:-1])
			labels_data.append(float_arr[-1:])
	else:
		print('File is not exist')
	return feature_data,labels_data

def train():
	x_data, y_data = load_data(file_path='customer_salary_satisfaction')
	x_train = x_data[:-200]
	y_train = y_data[:-200]
	lin_model = linear_model.LinearRegression()
	lin_model.fit(x_train, y_train)
	return lin_model

def test():
	x_data, y_data = load_data(file_path = 'customer_salary_satisfaction')
	x_test = x_data[-200:]
	y_test = y_data[-200:]
	lin_model = train()
	y_predict = lin_model.predict(x_test)
	square_enrror = np.mean((y_predict - y_test)**2)
	print(square_enrror)
	plt.scatter(x_test, y_test,color = 'black')
	plt.plot(x_test, y_predict,color = 'red')
	plt.show()

if(__name__ == '__main__'):
	test()
