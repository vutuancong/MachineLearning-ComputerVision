from sklearn.cluster import AffinityPropagation
# from utils import data_helpers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

__global_map  = map
lmap = lambda func,*iterable: list(__global_map(func,*iterable))
map = lmap

def read_feature_data(file_path):
    data = []
    if(os.path.isfile(file_path)):
        f = open(file_path,'r')
        samples = list(f.readlines())
        f.close()
        samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
        for sample in samples:
            sample_arr =  sample.split('\t')
            feature_arr = map(float,sample_arr)
            data.append(feature_arr)
    return data

def customer_clustering():
    data = read_feature_data(file_path='customer_data_min')
    aff_prop_model = AffinityPropagation(convergence_iter=250, max_iter=1000)
    aff_prop_model.fit(data)
    cluster_centers_indices = aff_prop_model.cluster_centers_indices_
    labels = aff_prop_model.labels_
    print(len(labels))  
    return data, labels

def get_data_by_label(feature_data, label_data, expect_label):
    feature_dim = len(feature_data[0])
    print(feature_dim)
    xs = []
    ys = []
    zs = []
    for sample_id, feature_vector in enumerate(feature_data):
        if label_data[sample_id] == expect_label:
            if feature_dim >=1:
                xs.append(feature_vector[0])
                print(feature_vector)
            if feature_dim >=2:
                ys.append(feature_vector[1])
            if feature_dim >=3:
                zs.append(feature_vector[2])
    return xs, ys, zs


def plot(data, labels):
    xs0, ys0, zs0 = get_data_by_label(data, labels, 0)
    xs1, ys1, zs1 = get_data_by_label(data, labels, 1)
    xs2, ys2, zs2 = get_data_by_label(data, labels, 2)
    xs3, ys3, zs3 = get_data_by_label(data, labels, 3)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs0, zs0, ys0, c='y', marker='o')
    ax.scatter(xs1, zs1, ys1, c='r', marker='o')
    ax.scatter(xs2, zs2, ys2, c='b', marker='o')
    ax.scatter(xs3, zs3, ys3, c='g', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def main():
    data, labels = customer_clustering()
    plot(data, labels)


if __name__ == '__main__':
    main()
