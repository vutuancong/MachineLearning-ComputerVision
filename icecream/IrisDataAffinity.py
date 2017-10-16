from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn import datasets

__global_map = map 
lmap = lambda func,*iterable : list(__global_map(func,*iterable))
map = lmap

def Input_and_set_data():
	iris_data = datasets.load_iris()
	data = iris_data.data
	labels = iris_data.target
	aff_pro_model = AffinityPropagation(convergence_iter  = 250, max_iter = 1000)
	aff_pro_model.fit(data)
	cluster_centers_indices = aff_pro_model.cluster_centers_indices_
	labels_model = aff_pro_model.labels_
	return data,labels

def get_data_by_label(data, labels, expect_label):
	xs = []
	ys = []
	zs = []
	for id_, data in enumerate(data):
		if(labels[id_] == expect_label):
			xs.append(data[0])
		if(labels[id_] == expect_label):
			ys.append(data[1])
		if(labels[id_] == expect_label):
			zs.append(data[2])
	return xs, ys, zs

def iris_plot(data, labels):
	xs0, ys0, zs0 = get_data_by_label(data, labels, 0)
	xs1, ys1, zs1 = get_data_by_label(data, labels, 1)
	xs2, ys2, zs2 = get_data_by_label(data, labels, 2)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(xs0, ys0, zs0, c = 'r', marker = "o")
	ax.scatter(xs1, ys1, zs1, c = 'b', marker = "o")
	ax.scatter(xs2, ys2, zs2, c = 'g', marker = "o")

	ax.set_xlabel("X_Label")
	ax.set_ylabel("Y_label")
	ax.set_zlabel("Z_label")
	plt.show()
def main():
	data, labels = Input_and_set_data()
	iris_plot(data, labels)

if __name__=='__main__':
	main()


