import json

import matplotlib.pyplot as plt

from TensorFlowUtils import TensorFlowUtils


class Drawing:

    def __init__(self):
        self.history = json.load(open(TensorFlowUtils.history_path, 'r'))

    plt.figure(figsize=(12, 7))

    def draw_accuracy_plot(self, y_limit=(0.8, 1.00)):
        plt.subplot(1, 1, 1)
        plt.plot(self.history['accuracy'], "r--")
        plt.plot(self.history['val_' + 'accuracy'], "g--")
        plt.ylabel("accuracy")
        plt.xlabel('Epoch')
        plt.ylim(y_limit)
        plt.legend(['train', 'test'], loc='best')
        plt.show()

    def draw_loss_plot(self, y_limit=(0.0, 1.0)):
        plt.subplot(1, 1, 1)
        plt.plot(self.history['loss'], "r--")
        plt.plot(self.history['val_' + 'loss'], "g--")
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.ylim(y_limit)
        plt.legend(['train', 'test'], loc='best')
        plt.show()
