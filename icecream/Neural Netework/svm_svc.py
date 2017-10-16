from sklearn.svm import SVC
from sklearn import datasets, metrics
import matplotlib.pyplot as plt


# def iris_classification():
#     iris = datasets.load_iris()
#     data = iris.data
#     labels = iris.target
#     svc_model = SVC()
#     svc_model.fit(data, labels)
#     predict_labels = svc_model.predict(data)
#     print(predict_labels)


def digit_classification():
    # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
    # License: BSD 3 clause
    digits = datasets.load_digits()

    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 3 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # pylab.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(4, 10, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('T: %i' % label)

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    
    print(":::::::::::::::::::::::::::::::::::::::::::::",n_samples)

    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples / 2:]
    predicted = classifier.predict(data[n_samples / 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:10]):
        plt.subplot(4, 10, index + 11)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('P: %i' % prediction)

    plt.show()


def main():
    print('starting')
    # iris_classification()
    digit_classification()


if __name__ == '__main__':
    main()