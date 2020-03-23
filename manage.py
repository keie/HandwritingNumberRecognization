import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28 images of handw-written digits0-9

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


#MODEL CREATION####################################
model=tf.keras.models.Secuential()
model.add(tf.keras.layers.Flatten())#inputLayer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))#dense layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))#outputLayer
#MODEL CREATION####################################

##TRAINING MODEL#####################################
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',

)
##TRAINING MODEL#####################################
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
#print(x_train[0])
