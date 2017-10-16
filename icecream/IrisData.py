from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
import unsupervise
from mpl_toolkits.mplot3d import Axes3D

def train():
	iris_data = datasets.load_iris()
	data = iris_data['data']
	labels = iris_data['target']
	kmeans_model = KMeans(n_clusters = 3)
	# kmeans_model = KNeighborsClassifier()
	# kmeans_model.fit(data,labels)
	kmeans_model.fit(data)
	centroids = kmeans_model.cluster_centers_
	print(centroids)
	fw =  open('kmeans_model.pkl','wb')
	pickle.dump(kmeans_model,fw)
	return kmeans_model,data, labels

def test():
	kmeans_model, data, labels = train()
	predict_labels = kmeans_model.predict(data)
	# correct_count =  0
	# total = 150
	# check_label_array = [[0,0,0],[0,0,0],[0,0,0]]
	# for id_, label_ in enumerate(predict_labels):
	# 	if id_ < 50:
	# 		check_label_array[0][label_]+=1
	# 	elif id_ <100:
	# 		check_label_array[1][label_]+=1
	# 	else :
	# 		check_label_array[2][label_]+=1
	# print(check_label_array)
	# for i in range(3):
	# 	correct_count += max(check_label_array[i])
	# print(float(correct_count) / total)
	return data, predict_labels

def iris_plot(data, labels):
	xs0, ys0, zs0 = unsupervise.get_data_by_label(data, labels, 0)
	xs1, ys1, zs1 = unsupervise.get_data_by_label(data, labels, 1)
	xs2, ys2, zs2 = unsupervise.get_data_by_label(data, labels, 2)
	# print(xs0,ys0, zs0)
	fig  = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(xs0, ys0, zs0, c = 'b')
	ax.scatter(xs1, ys1, zs1, c = 'r')
	ax.scatter(xs2, ys2, zs2, c = 'g')
	plt.show()

def main():
	data,labels = test()
	iris_plot(data,labels)


if __name__ == '__main__':
	main()
