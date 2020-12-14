import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

from sources.DataLoader import DataLoader
from sources.Drawing import Drawing


class TensorFlowUtils:

    def __init__(self):
        self.dataLoader = DataLoader()
        self.__prepareModel()

    def __prepareModel(self):
        number_of_headers = self.dataLoader.get_column_number()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(number_of_headers, input_shape=(number_of_headers,)),
            tf.keras.layers.Dense(120, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(len(self.dataLoader.class_names), activation=tf.keras.activations.softmax)
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # kończenie uczenia gdy strata na zbiorze testowym nie rośnie
        earlyStop = EarlyStopping(monitor='val_loss',
                                  patience=3,
                                  verbose=1)
        history = model.fit(self.dataLoader.get_training_data(), self.__get_prepared_training_labels(),
                            epochs=50,
                            verbose=1,
                            batch_size=256,
                            validation_data=(self.dataLoader.get_test_data(), self.__get_prepared_test_labels()),
                            callbacks=[earlyStop]
                            )
        Drawing().draw_curves(history)

    def __get_prepared_training_labels(self):
        training_labels = self.dataLoader.get_training_labels()
        data = tf.keras.utils.to_categorical(training_labels, len(self.dataLoader.class_names))
        print("prepared training labels: ")
        print(data)
        return data

    def __get_prepared_test_labels(self):
        test_labels = self.dataLoader.get_test_labels()
        data = tf.keras.utils.to_categorical(test_labels, len(self.dataLoader.class_names))
        print("prepared test labels: ")
        print(data)
        return data
