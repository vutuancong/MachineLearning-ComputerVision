import matplotlib.pyplot as plt 
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise = 0.20)
# plt.scatter(X[:,0], X[:,1], s = 40, c = y, cmap = plt.cm.Spectral)
# plt.show();

# def plot_decision_boundary(pred_func):
logreg = linear_model.LogisticRegression(C = 1e5)
logreg.fit(X,y)

x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))
Z = logreg.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
plt.scatter(X[:,0],X[:,1], c = y, cmap = plt.cm.Spectral)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)

plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plot.show()

# plot_decision_boundary()