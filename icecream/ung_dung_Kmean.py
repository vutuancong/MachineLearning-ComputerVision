import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import os


def color_quantization(n_colors = 128, file_path = ''):
	image = None
	#check file and input
	if(len(file_path) > 0 and (os.path.isfile(file_path))):
		image = mpimg.imread(file_path)
	
	image = np.array(image, dtype = np.float64) / 255
	w, h, d = original_shape = tuple(image.shape)
	assert d == 3
	image_array = np.reshape(image,(w*h, d))
	print(image_array)
	image_array_sample = shuffle(image_array, random_state = 0)[:1000]

	#using Kmean
	kmeans = KMeans(n_clusters = n_colors, random_state = 0).fit(image_array_sample)
	labels = kmeans.predict(image_array)
	centroids = kmeans.cluster_centers_
	# print(labels)
	# print(centroids)

	#using random color
	codebook_random = shuffle(image_array,random_state = 0)[:n_colors+1]
	labels_random = pairwise_distances_argmin(codebook_random,image_array,axis = 0)

	return [kmeans, image, labels, codebook_random, labels_random, w, h]

def recreate_image(codebook, labels, w, h):
	d = codebook.shape[1]
	image = np.zeros((w, h, d))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image[i][j] = codebook[labels[label_idx]]
			label_idx += 1
	return image

def plot_image(kmeans, image, labels, codebook_random, labels_random, w, h):
	plt.figure(1)
	plt.clf()
	ax = plt.axes([0, 0, 1, 1])
	plt.axis('off')
	plt.title('Original image ')
	plt.imshow(image)

	plt.figure(2)
	plt.clf()
	ax = plt.axes([0, 0, 1, 1])
	plt.axis('off')
	plt.title('Image 64 color, Kmean')
	plt.imshow(recreate_image(kmeans.cluster_centers_,labels, w, h))

	plt.figure(3)
	plt.clf()
	ax = plt.axes([0, 0, 1, 1])
	plt.axis('off')
	plt.title('Color Randon')
	plt.imshow(recreate_image(codebook_random, labels_random, w, h))
	plt.show()

def main():
	kmeans, image, labels, codebook_random, labels_random, w, h = color_quantization(file_path = "nha_trang.jpg")
	plot_image(kmeans, image, labels, codebook_random, labels_	

if __name__ == '__main__':
	main()
