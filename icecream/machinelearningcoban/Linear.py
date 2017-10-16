import numpy as np
import matplotlib.pyplot as plt 

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


# ex X = [[1,2,3,1],[2,3,4,5],]  X(2,4)
# X.shape = (2,4); X.shape[0] = 2

one = np.ones((X.shape[0],1)) 

# a = np.array([[1,2],[3,4]])
# b = np# np.concatenate((a,b), axis = 0)
# array([[1,2],
# 	   [3,4],
# 	   [5,6]])

# np.concatenate((a,b),axis = 1)
# array([[1,2,5],
# 	   [3,4,6]])

Xbar = np.concatenate((one,X), axis = 1)
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T,y)

# np.linalg.pinv la gia nghich dao .
# ex: A+ = np.linalg.pinv(A)

w = np.dot(np.linalg.pinv(A),b)

print('w = ', w)

w_0 = w[0][0]
w_1 = w[1][0]
#linespace x0 = np.linspace la trong khoang 145 185 lay ra n so co khoang cach la const

x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print('predict weight of preson with height 155 cn: %.2f (kg), real number: 52(kg)' %(y1))
print('predict weight of person with height 160 cm: %.2f (kg), real number: 56(kg)' %(y2))

plt.plot(X.T,y.T,'ro')
plt.plot(x0,y0)
plt.axis([140,190,45,75])
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()
