import numpy as np
import pandas as pd
from sklearn import preprocessing

np.set_printoptions(precision=3, suppress=True)


class DataLoader:
    trainingDataPercentage = 70
    class_names = ['jadalny', 'niejadalny']

    def __init__(self):
        shuffled_data = pd \
            .read_csv("../grzyby.csv", sep=";", header=0) \
            .sample(frac=1) \
            .reset_index(drop=True)
        # TODO("zmiana na wprowadzeie przez uzytkownika")
        self.labels = np.array(shuffled_data.pop("class"))
        self.all_data = np.array(self.__min_max_normalization(shuffled_data))
        print("labels")
        print(self.labels)
        print("all data")
        print(self.all_data)

    def __min_max_normalization(self, data):
        return preprocessing.normalize(data)

    def __get_training_data_rows_number(self):
        data_size = len(self.all_data)
        return int(data_size * (self.trainingDataPercentage / 100.0))

    def get_training_data(self):
        data = self.all_data[0:self.__get_training_data_rows_number()]
        print("training data size: " + str(len(data)))
        print(data)
        return data

    def get_training_labels(self):
        data = self.labels[0:self.__get_training_data_rows_number()]
        print("training labels size: " + str(len(data)))
        print(data)
        return data

    def get_test_data(self):
        data_size = len(self.all_data)
        training_rows_size = self.__get_training_data_rows_number()
        data = self.all_data[-(data_size - training_rows_size):]
        print("test data size: " + str(len(data)))
        print(data)
        return data

    def get_test_labels(self):
        data_size = len(self.labels)
        training_rows_size = self.__get_training_data_rows_number()
        data = self.labels[-(data_size - training_rows_size):]
        print("test labels size: " + str(len(data)))
        print(data)
        return data

    def get_column_number(self):
        headersSize = self.all_data.shape[1]
        print("number of headers: " + str(headersSize))
        return headersSize
