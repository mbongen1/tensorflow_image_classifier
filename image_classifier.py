
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

