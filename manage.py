import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28 images of handw-written digits0-9

(x_train,y_train),(x_test,y_test) = mnist.load_data()
plt.imshow(x_train[0])
plt.show()
#print(x_train[0])
