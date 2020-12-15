import json

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping


class TensorFlowUtils:
    checkpoint_path = "model/model.ckpt"
    history_path = "model/model_history.json"

    def __init__(self, data_loader):
        self.dataLoader = data_loader

    def prepareModel(self):
        number_of_headers = self.dataLoader.get_column_number()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(number_of_headers, input_shape=(number_of_headers,)),
            tf.keras.layers.Dense(120, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(len(self.dataLoader.class_names), activation=tf.keras.activations.softmax)
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self, model, epochs=50, batch_size=256):
        # kończenie uczenia gdy strata na zbiorze testowym nie rośnie
        earlyStop = EarlyStopping(monitor='val_loss',
                                  patience=3,
                                  verbose=1)
        # zapis stanu modelu po zakończonym uczeniu
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = model.fit(self.dataLoader.get_training_data(), self.__get_prepared_training_labels(),
                            epochs=epochs,
                            verbose=1,
                            batch_size=batch_size,
                            validation_data=(self.dataLoader.get_test_data(), self.__get_prepared_test_labels()),
                            callbacks=[earlyStop, cp_callback]
                            )
        json.dump(history.history, open(self.history_path, 'w'))
        return history

    def load_saved_weights(self, model):
        model.load_weights(self.checkpoint_path)

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

    def predict(self, model):
        predicted_data = model.predict(self.dataLoader.get_all_loaded_data())
        return predicted_data
