import pandas as pd 
import quandl
import math
import numpy as np 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style


def load_data():
	style.use('ggplot')

	df = quandl.get("CBOE/TYX")
	df = df [['Open','High','Low','Close']]

	df['LowHigh'] =  (df['High'] - df['Low']) / df['Low'] *100
	df['OpenClose'] = (df['Close'] - df['Open'])/df['Open'] *100
	df  = df[['Open','High','Low','Close','LowHigh','OpenClose']]
	forecast_col = 'Low'

	#lap day thuoc tinh du lieu
	df.fillna(-99999, inplace = True)
	
	# math.ceil(x) tra ve chieu cao cua x - gia tri so nguyen nho nhat lon hon hoac  = x;
	# forecast_out trich xuat 1 phan do dai du lieu 
	forecast_out = int(math.ceil(0.01*len(df)))
	
	#shuffle data trong khoang -forecast_out
	df['label'] = df[forecast_col].shift(-forecast_out)
	print(df['label'])
	# print(df.head())
	
	X = np.array(df.drop(['label'],1)) # tra ve object moi
	print(X)
	X = preprocessing.scale(X)
	print(X)
	# X_data_train = X[:-forecast_out]
	# Y_label = X[-forecast_out:]

	# df.dropna(inplace =  True)

	# model = LinearRegression()
	# model.fit(X_data_train,Y_data_test)
	# accuracy = model.score()
	# print(accuracy)
	# print(df[['Low']])

load_data()
