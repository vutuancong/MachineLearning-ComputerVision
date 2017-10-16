import os
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

__global_map = map
lmap = lambda func,*iterable:list(__global_map(func,*iterable))
map = lmap

def read_feature_data(file_path):
	data = []
	if os.path.isfile(file_path):
		f = open(file_path,'r')
		samples = list(f.readlines())
		f.close()
		samples = [sample.strip() for sample in samples if(len(sample.strip()))>0]
		for sample in samples:
			sample_arr = sample.split('\t')
			feature_arr = map(float,sample_arr)
			data.append(feature_arr)
	return data

def user_rating_clustering():
	data = read_feature_data(file_path = 'customer_data')
	ms_model = MeanShift()
	predict_labels = ms_model.fit_predict(data)
	cluster_center_indices = ms_model.cluster_centers_

	print(predict_labels)

	return predict_labels

def Input_and_Data_set():
	data = read_feature_data(file_path = 'customer_data')
	ms_model = MeanShift(bandwidth = 0.18)

	predict_labels = ms_model.fit_predict(data)
	cluster_center_indices = ms_model.cluster_centers_
	total_guess_num = 800
	correct_guess_num = 0
	predict_labels_count_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
	for id_, label_ in enumerate(predict_labels):
		if id_ < 200:
			predict_labels_count_matrix[0][label_] +=1
		elif id_ < 400:
			predict_labels_count_matrix[1][label_] +=1
		elif id_ < 600:
			predict_labels_count_matrix[2][label_] +=1
		else:
			predict_labels_count_matrix[3][label_] +=1
	for i in range(4):
		correct_guess_num += max(predict_labels_count_matrix[i])
	accuary = correct_guess_num/total_guess_num
	print('accuary : ' ,accuary)
	print(predict_labels)
	print("cluseter center 1->4: ")
	print(cluster_center_indices)
	return data, predict_labels

def get_data_by_label(feature_data, label_data, expect_label):
	feature_dim = len(feature_data[0])
	xs = []
	ys = []
	zs = []
	for id_, data in enumerate(feature_data):
		if label_data[id_] == expect_label:
			if feature_dim >=1:
				xs.append(data[0])
				print(data[0])
			if feature_dim >=2:
				ys.append(data[1])				
			if feature_dim >=3:
				zs.append(data[2])
	return xs, ys, zs

def plot(data, labels):
	# print(data,labels)
	xs0, ys0, zs0 = get_data_by_label(data, labels, 0)
	xs1, ys1, zs1 = get_data_by_label(data, labels, 1)
	xs2, ys2, zs2 = get_data_by_label(data, labels, 2)
	xs3, ys3, zs3 = get_data_by_label(data, labels, 3)

	fig = plt.figure()

	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(xs0, ys0, zs0, c = 'r', marker = "o")
	ax.scatter(xs1, ys1, zs1, c = 'y', marker = "o")
	ax.scatter(xs2, ys2, zs2, c = 'g', marker = "o")
	ax.scatter(xs3, ys3, zs3, c = 'b', marker = "o")

	ax.set_xlabel("X label")
	ax.set_ylabel("Y label")
	ax.set_zlabel("Z label")
	plt.show()

def main():
	print('Staring..')
	user_rating_clustering()
	data, labels  = Input_and_Data_set()
	plot(data, labels)

if __name__=="__main__":
	main()
