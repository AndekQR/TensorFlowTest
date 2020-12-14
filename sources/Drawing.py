import matplotlib.pyplot as plt


class Drawing:
    def draw_curves(self, history, key1='accuracy', ylim1=(0.8, 1.00), key2='loss', ylim2=(0.0, 1.0)):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history[key1], "r--")
        plt.plot(history.history['val_' + key1], "g--")
        plt.ylabel(key1)
        plt.xlabel('Epoch')
        plt.ylim(ylim1)
        plt.legend(['train', 'test'], loc='best')

        plt.subplot(1, 2, 2)
        plt.plot(history.history[key2], "r--")
        plt.plot(history.history['val_' + key2], "g--")
        plt.ylabel(key2)
        plt.xlabel('Epoch')
        plt.ylim(ylim2)
        plt.legend(['train', 'test'], loc='best')

        plt.show()
