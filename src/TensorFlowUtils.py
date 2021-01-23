import json

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

from Algorithm import Algorithm


class TensorFlowUtils:
    checkpoint_path = "model/model.ckpt"
    history_path = "model/model_history.json"

    def __init__(self, data_loader):
        self.dataLoader = data_loader

    def prepare_model(self, algorithm: Algorithm, learning_rate=0.001) -> tf.keras.models.Sequential:
        number_of_headers = self.dataLoader.get_column_number()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(number_of_headers, input_shape=(number_of_headers,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(len(self.dataLoader.class_names), activation=tf.keras.activations.softmax)
        ])

        if algorithm == Algorithm.adaptive_moment_estimation:
            self.adam_algorithm(learning_rate, model)
        else:
            self.lovenberg_marquardt_algorithm(learning_rate, model)

        model.summary()
        return model

    # algorytm sgd wykorzystujący metodą gradientową
    def lovenberg_marquardt_algorithm(self, learning_rate,
                                      model: tf.keras.models.Sequential):
        my_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # metoda spadku gradientu
    def adam_algorithm(self, learning_rate, model: tf.keras.models.Sequential):
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, model: tf.keras.models.Sequential, epochs=50, batch_size=32):
        # kończenie uczenia gdy strata na zbiorze testowym nie poprawia się
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1)
        # zapis stanu modelu po zakończonym uczeniu
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=1)

        history = model.fit(self.dataLoader.get_training_data(), self.__get_prepared_training_labels(),
                            epochs=epochs,
                            verbose=1,
                            batch_size=batch_size,
                            validation_data=(self.dataLoader.get_test_data(), self.__get_prepared_test_labels()),
                            callbacks=[early_stop, cp_callback]
                            )
        json.dump(history.history, open(self.history_path, 'w'))
        return history

    def load_saved_weights(self, model: tf.keras.models.Sequential):
        try:
            model.load_weights(self.checkpoint_path)
        except tf.errors.NotFoundError:
            self.train_model(model)

    def __get_prepared_training_labels(self):
        training_labels = self.dataLoader.get_training_labels()

        data = tf.keras.utils.to_categorical(training_labels, len(self.dataLoader.class_names))

        return data

    def __get_prepared_test_labels(self):
        test_labels = self.dataLoader.get_test_labels()
        data = tf.keras.utils.to_categorical(test_labels, len(self.dataLoader.class_names))

        return data

    def predict(self, model: tf.keras.models.Sequential):
        predicted_data = model.predict(self.dataLoader.get_all_loaded_data())
        return predicted_data
