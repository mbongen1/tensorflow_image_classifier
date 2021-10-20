
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
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#The next step is to compile the model which defines some of the options to be used during training (eg. the loss function used, the metrics to be reported etc.)
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#Training the model
model.fit(train_images, train_labels, epochs=10)

#Evaluate how the model performs on the test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#Making predictions
#First, we add a Softmax layer to convert logits to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#Here, we make make predictions for all of the images in the test set and print the model's confidence for the first predicted image
predictions = probability_model.predict(test_images)
print("The model's prediction for the first image (as a series of probabilites) is: ", predictions[0])
max = np.argmax(predictions[0]) #The model is most confident in this label out of the 10 possibilities from 0 to 9
print("The model's prediction for the correct label of the image is: ", class_names[max])


#Inserting some functions from the Tensorflow Docs for displaying predictions
def plot_image(i, predictions_array, true_label, img):
      true_label, img = true_label[i], img[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      
      plt.imshow(img, cmap=plt.cm.binary)
      
      predicted_label = np.argmax(predictions_array)
      if predicted_label == true_label:
          color = 'blue'
      else:
          color = 'red'
          
      plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

