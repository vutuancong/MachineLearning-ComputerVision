import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import os

def color_quatization(n_colors = 64, file_path = ""):
	image = None
	if(len(file_path)>0 and (os.path.isfile(file_path))):
		image = mpimg.imread(file_path)

	image = np.array(image, dtype = np.float64) / 255
	w, h, d = original_shape = tuple(image.shape)
	assert d == 3
	image_array = np.reshape(image,(w*h, d))
	image_array_sample = shuffle(image_array, random_state = 0)[:1000]

	kmean = KMeans(n_clusters = n_colors, random_state = 0).fit(image_array_sample)
	labels = kmean.predict(image_array)
	centroids = kmean.cluster_centers_

	return kmean, image, w, h, centroids, labels

def recreate_image(codebook, labels, w, h):
	d = codebook.shape[1]
	image = np.zeros((w, h, d))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image[i][j] = codebook[labels[label_idx]]
			label_idx += 1
	return image
def  plot_image(kmeans, image, w, h, centroids, labels):
	plt.figure(1)
	plt.clf()
	ax = plt.axes([0, 0, 1, 1])
	plt.axis('off')
	plt.title('Original image')
	plt.imshow(image)

	plt.figure(2)
	plt.clf()
	ax = plt.axes([0, 0, 1, 1])
	plt.axis('off')
	plt.title('Image 64 color, Kmean')
	plt.imshow(recreate_image(centroids,labels, w, h))

	plt.show()
def main():
	kmeans, image, w, h, centroids, labels = color_quatization(file_path = 'nha_trang.jpg')
	plot_image(kmeans, image, w, h, centroids,labels)

if __name__ =='__main__':
	main()