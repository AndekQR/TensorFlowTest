# wczytanie potrzebnych bibliotek
import tensorflow as tf
import matplotlib.pyplot as plt

#wczytanie danych 
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()

print(f'Zbiór uczący: {X_train.shape}, zbiór walidacyjny: {X_val.shape}')

# plt.figure(figsize=(7,7))
# plt.imshow(X_train[0], cmap=plt.cm.binary)
# plt.colorbar()
# plt.show()
#
# def plot_digit(digit, dem=28, font_size=8):
#     max_ax = font_size * dem
#
#     fig = plt.figure(figsize=(10,10))
#     plt.xlim([0, max_ax])
#     plt.ylim([0, max_ax])
#     plt.axis('off')
#
#     for idx in range(dem):
#         for jdx in range(dem):
#             t = plt.text(idx*font_size, max_ax - jdx*font_size,
#                          digit[jdx][idx], fontsize=font_size,
#                          color="#000000")
#             c = digit[jdx][idx] / 255.
#             t.set_bbox(dict(facecolor=(c, c, c), alpha=0.5,
#                             edgecolor='#f1f1f1'))
#
#     plt.show()
#
# plot_digit(X_train[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(14,10))
for i in range(40):
    plt.subplot(5, 8, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()