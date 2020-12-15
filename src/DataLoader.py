import numpy as np
import pandas as pd
from sklearn import preprocessing

np.set_printoptions(precision=3, suppress=True)


class DataLoader:
    trainingDataPercentage = 70
    class_names = ['jadalny', 'niejadalny']

    def __init__(self, data_path, shuffle_data=True):
        shuffled_data = pd \
            .read_csv(data_path, sep=";", header=0)

        if shuffle_data:
            shuffled_data = shuffled_data \
                .sample(frac=1) \
                .reset_index(drop=True)

        # TODO("zmiana na wprowadzeie przez uzytkownika")
        if 'class' in shuffled_data.columns:
            self.labels = np.array(shuffled_data.pop("class"))
        else:
            self.labels = []
        self.all_data = np.array(self.__min_max_normalization(shuffled_data))


    def __min_max_normalization(self, data):
        return preprocessing.normalize(data)


    def __get_training_data_rows_number(self):
        data_size = len(self.all_data)
        return int(data_size * (self.trainingDataPercentage / 100.0))


    def get_training_data(self):
        data = self.all_data[0:self.__get_training_data_rows_number()]
        # print("training data size: " + str(len(data)))
        # print(data)
        return data


    def get_training_labels(self):
        data = self.labels[0:self.__get_training_data_rows_number()]
        # print("training labels size: " + str(len(data)))
        # print(data)
        return data


    def get_test_data(self):
        data_size = len(self.all_data)
        training_rows_size = self.__get_training_data_rows_number()
        data = self.all_data[-(data_size - training_rows_size):]
        # print("test data size: " + str(len(data)))
        # print(data)
        return data


    def get_test_labels(self):
        data_size = len(self.labels)
        training_rows_size = self.__get_training_data_rows_number()
        data = self.labels[-(data_size - training_rows_size):]
        # print("test labels size: " + str(len(data)))
        # print(data)
        return data


    def get_column_number(self):
        headersSize = self.all_data.shape[1]
        # print("number of headers: " + str(headersSize))
        return headersSize


    def get_all_loaded_data(self):
        """wraca wszystkie wczytane dane bez kolumny klas"""
        return self.all_data


    def get_all_loaded_labels(self):
        """zwraca wszytkie wczytane klasy"""
        return self.labels


    def append_predict_column(self, file_path, data):
        flat_array = [item for sublist in data for item in sublist]
        df = pd.read_csv(file_path)
        new_column = pd.DataFrame({'predicted': flat_array})
        df = df.merge(new_column, left_index=True, right_index=True)
        df.to_csv(file_path, index=False)
