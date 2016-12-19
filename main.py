import numpy as np
import pandas as pa

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sk_learn_classifiers import SKClassifiers

# import tensorflow as tf
## settings
LEARNING_RATE = 1e-4
## set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500
DROPOUT = 0.5
BATCH_SIZE = 50
## set to 0 to train on all available data
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10

print("*****************S T A R T***************")
train_data = pa.read_csv('data/train.csv')
print("Train data processing ended")

## split train_data into arrays of 784 digits. 784 = 28*28. images contains images as array of digits
images = train_data.iloc[:, 1:].values
## number of images is training data set
print("each image contains %s digits", len(images))
images = images.astype(np.float)

## convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

SKClassifiers.random_forest(train_data)
SKClassifiers.desicion_tree(train_data)
# SKClassifiers.k_neighbors(train_data)

image_size = images.shape[1]

## in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

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
