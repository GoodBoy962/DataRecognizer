import numpy as np
import pandas as pa

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

## settings
LEARNING_RATE = 1e-4
## set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

## set to 0 to train on all available data
VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 10

train_data = pa.read_csv('data/train.csv')
print("rows number: %s || columns number: %s" % train_data.shape)

## split train_data into arrays of 784 digits. 784 = 28*28. images contains images as array of digits
images = train_data.iloc[:, 1:].values
## number of images is training data set
print("each image contains %s digits", len(images))
images = images.astype(np.float)

## convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]

## in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


## display image
def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)


# display(images[IMAGE_TO_DISPLAY])

## labels of train data
train_data_labels = train_data[[0]].values.ravel()


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels = dense_to_one_hot(train_data_labels, 10)
labels = labels.astype(np.uint8)

# split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# # weight initialization
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(train_data.iloc[:, 1:].values, train_data[[0]].values.ravel())
pred = classifier.predict(pa.read_csv("data/test.csv"))

np.savetxt('submission_rand_forest.csv', np.c_[range(1, len(pa.read_csv("data/test.csv")) + 1), pred],
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')
