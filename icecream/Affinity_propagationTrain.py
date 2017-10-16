from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

__global_map = map
lmap = lambda func,*iterable:list(__global_map(func,*iterable))
map = lmap

def read_feature_data(file_path):
	data = []
	if(os.path.isfile(file_path)):
		f = open(file_path,'r')
		samples = list(f.readlines())
		f.close()
		samples = [sample.strip() for sample in samples if(len(sample.strip())) > 0]
		for sample in samples:
			sample_arr = sample.split("\t")
			feature_arr = map(float,sample_arr)
			data.append(feature_arr)

	return data

def customer_clustering():
	data = read_feature_data(file_path = 'customer_data_min')
	aff_propa_model =  AffinityPropagation(convergence_iter = 250,max_iter = 1000)
	aff_propa_model.fit(data)
	cluster_centre_indices = aff_propa_model.cluster_centers_indices_
	labels = aff_propa_model.labels_

	print(labels)

	return data,labels

def get_data_by_label(feature_data,label_data,expect_label):
	feature_dim = len(feature_data[0])
	xs = []
	ys = []
	zs = []
	for label_id, data_id in enumerate(feature_data):
		if label_data[label_id] == expect_label:
			if(feature_dim >=1):
				xs.append(data_id[0])
			if(feature_dim >=2):
				ys.append(data_id[1])
			if(feature_dim >=3):
				zs.append(data_id[2])
	return xs, ys, zs

def plot(data, label):
	xs0, ys0, zs0 = get_data_by_label(data,label,0)
	xs1, ys1, zs1 = get_data_by_label(data,label,1)
	xs2, ys2, zs2 = get_data_by_label(data,label,2)
	xs3, ys3, zs3 = get_data_by_label(data,label,3)
	fig = plt.figure()

	ax = fig.add_subplot(111,projection = '3d')
	ax.scatter(xs0, ys0, zs0, c = 'y',marker = 'o')
	ax.scatter(xs1, ys1, zs1, c = 'r',marker = 'o')
	ax.scatter(xs2, ys2, zs2, c = 'b',marker = 'o')
	ax.scatter(xs3, ys3, zs3, c = 'g',marker = 'o')

	ax.set_xlabel('X label')
	ax.set_ylabel('Y label')
	ax.set_zlabel('Z label')

	plt.show()

def main():
	data, label = customer_clustering()
	plot(data,label)

if __name__ == '__main__':
	main()

