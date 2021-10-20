# **Introductory Image Classifier**

This program was set up by following the [Tensorflow "Basic Image Classification" tutorial](https://www.tensorflow.org/tutorials/keras/classification). Slight tweaks and adjustments were made for outputting the results of the model.



## **What It Does**
At a high level, the program uses Tensorflow to create a model for classifying images of articles of clothing.



## **How It Does It**
The prorgam makes use of the Fashion MNIST dataset and splits the 70 000 images into 60 000 training images and 10 000 test images. Each image is originally represented by a 28x28 matrix of pixels with values ranging from 0 to 255. This range is transformed to a range of 0 to 1. A Neural Network is then modelled and fitted using Keras. The accuracy metric is printed for each epoch during the training process. The model then makes predictions on the 10 000 test images. Lastly, the program ouputs the results from the model's first 9 test predictions in a nicely colored 3x3 panel. Each entry in the panel represents the model's confidence levels for the prediction along with the correct label in brackets. For example: "Sneaker 94% (Sneaker)".



## **Set Up and Usage**
- Install Tensorflow (`pip install tensorflow`)
- Run `python image_classifier.py`
- Lastly, let the program do its thing. 
