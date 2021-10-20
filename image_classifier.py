
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#These print out some infomration about the training set (which should be 60 000 images of 28x28 pixels) and the 10 000 labels for the training set
#print(train_images.shape)
#print(len(train_labels))
#print(train_labels)

#Pixel values range from 0 to 255 as shown below by the code below
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(True)
#plt.show()

#For better modelling, we scale the values to be between 0 and 1 for the training set and the test set
train_images = train_images / 255.0
test_images = test_images / 255.0

#This part of the code displays the first 25 images in our transformed training set with the label below each image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#This segment builds the model (it's not yet trained or anything but the initial structure is set up). The structure is a Neural Network with 784(=28x28) input nodes, 10 output nodes with 1 hidden layer of 128 nodes. Is this optimal? I don't know. This is simply the suggested structure for this dataset in the Tensorflow Docs.
model = tf.keras.model.Sequential([
    tf.keras.layers.Flatter(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#The next step is to compile the model which defines some of the options to be used during training (eg. the loss function used, the metrics to be reported etc.)
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True), metrics=['accuracy'])

#Training the model
model.fit(train_images, train_labels, epochs=10)

