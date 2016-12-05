import numpy as np
import pandas as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time


def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__

    return wrapper


class SKClassifiers:
    def __init__(self):
        super().__init__()

    @staticmethod
    def desicion_tree(train_data):
        # use DecisionTree to fit and predict
        classifier = DecisionTreeClassifier()
        start_time = time.time()
        classifier.fit(train_data.iloc[:, 1:].values, train_data[[0]].values.ravel())
        pred = classifier.predict(pa.read_csv("data/test.csv"))
        print("Decision Tree: ", time.time() - start_time)

        np.savetxt('sk_submissions/submission_decision_tree.csv',
                   np.c_[range(1, len(pa.read_csv("data/test.csv")) + 1), pred],
                   delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    @staticmethod
    def random_forest(train_data):
        # use RandomForestClassifier to fit and predict
        classifier = RandomForestClassifier(n_estimators=100)
        start_time = time.time()
        classifier.fit(train_data.iloc[:, 1:].values, train_data[[0]].values.ravel())
        pred = classifier.predict(pa.read_csv("data/test.csv"))
        print("Random Forest: ", time.time() - start_time)

        np.savetxt('sk_submissions/submission_rand_forest.csv',
                   np.c_[range(1, len(pa.read_csv("data/test.csv")) + 1), pred],
                   delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    @staticmethod
    def k_neighbors(train_data):
        # use KNeighborsClassifier to fit and predict
        classifier = KNeighborsClassifier()
        start_time = time.time()
        classifier.fit(train_data.iloc[:, 1:].values, train_data[[0]].values.ravel())
        pred = classifier.predict(pa.read_csv("data/test.csv"))
        print("KNeighbors: ", time.time() - start_time)

        np.savetxt('sk_submissions/submission_k_neigbours.csv',
                   np.c_[range(1, len(pa.read_csv("data/test.csv")) + 1), pred],
                   delimiter=',', header='ImageId,Label', comments='', fmt='%d')
