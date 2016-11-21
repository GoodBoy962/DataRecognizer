import numpy as np
import pandas as pa

class DataPreparator:

    def __init__(self):
        super().__init__()

    def prepare_data(self, path):
        train_data = pa.read_csv('data/train.csv')
        print("|| rows number: %s || columns number: %s" % train_data.shape)