import numpy as np
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pa
import time

print("*****************S T A R T***************")
train_data = pa.read_csv('data/train.csv')
print("Train data processing ended")

test_idx = []


def split_data(test_idx):
    for i in range(len(train_data[[0]].values.ravel())):
        if i % 2 != 0:
            test_idx.append(i)


split_data(test_idx)

target = train_data[[0]].values.ravel()
values = train_data.iloc[:, 1:].values

train_target = np.delete(target, test_idx)
train_data = np.delete(values, test_idx, axis=0)

test_target = target[test_idx]
test_data = values[test_idx]

# clf1 = neighbors.KNeighborsClassifier()
# clf1.fit(train_data, train_target)

clf2 = tree.DecisionTreeClassifier()
clf2.fit(train_data, train_target)

clf3 = ensemble.RandomForestClassifier()
clf3.fit(train_data, train_target)

# start_time = time.time()
# clf1_prediction = clf1.predict(test_data)
# print("KNeighbors classifier: {}".format(time.time() - start_time))

# start_time = time.time()
clf2_prediction = clf2.predict(test_data)
# print("Bayes classifier: {}".format(time.time() - start_time))

# start_time = time.time()
clf3_prediction = clf3.predict(test_data)
# print("Random forest classifier: {}".format(time.time() - start_time))

# print(accuracy_score(test_target, clf1_prediction))
print(accuracy_score(test_target, clf2_prediction))
print(accuracy_score(test_target, clf3_prediction))

# print(recall_score(test_target, clf1_prediction, average=None))
print(recall_score(test_target, clf2_prediction, average='weighted'))
print(recall_score(test_target, clf3_prediction, average='micro'))

# print(precision_score(test_target, clf1_prediction))
print(precision_score(test_target, clf2_prediction, average='micro'))
print(precision_score(test_target, clf3_prediction, average='micro'))
