import numpy as np 
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 

X = np.array([[147,150,153,158,163,165,168,170,173,175,178,180,183]]).T
y = np.array([[49,50,51,54,58,59,60,62,63,64,66,67,68]]).T



one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X),axis = 1)

regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(Xbar,y)
# print(regr.coef_)

w0 = regr.coef_[0][:-1]
w1 = regr.coef_[0][-1:]

x0 = np.linspace(145,185,2)
y0 = w0 + w1*x0

plt.plot(X,y,'ro')
plt.plot(x0,y0)
plt.axis([140,190,45,75])
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()
