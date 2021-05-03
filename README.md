# [Python] CNN to detect if mask is being worn

## Aim 
The aim of this project is to solve the problem of binary image classification using a class of deep learning neural networks CNN. It is the most established algorithm among various deep learning models, a class of artificial neural networks that has been a dominant method in computer vision tasks.
The chosen dataset is Facemask dataset. It is highly relatable of nowadays situation all over the world COVID-19. As per WHO, face masks combined with other preventive measures such as frequent hand-washing and social distancing help slow down the spread of the coronavirus.
Image classification is the task of assigning the input image one of the labels from already fixed set of categories. It is a complex procedure which relies on many different components. 

## Dataset

The [chosen dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) contains 11 792 RGB images of people faces with or without the masks, of which
10 000 are for training purposes, 800 for validation purposes and 992 for testing purposes to see how
well our model performs. All the images are in png format and of different sizes.

## Model
### Convolutional Neural Network
Convolutional Neural Networks are made up of neurons that have learnable weights and biases. Each
neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity.
The whole network still expresses a single differentiable score function: from the raw image pixels on
one end to class scores at the other. And they still have a loss function on the last layer.
### Results
This project evaluated different models and hyperparameters on the facemask dataset. All images
were transformed to a size of 128 x 128, where every image was downsized. Every CNN was trained
using 30 and 50 epochs as these values were determined by exploratory experiments that evaluated
validation and test scores every epoch.
To determine how well our model trained we will use validation curves. They display a hyperparam-
eter, in our case, the training epoch on horizontal axis and quality metric on the vertical axis. Our
quality metrics are accuracy and loss. Epochs are used to determine if longer training improves our
model's performance. By plotting loss and accuracy we can see if overfitting becomes the problem
and where.
Loss function, in our case Binary Cross-Entropy loss, measures the performance of a classification
model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the
predicted probability diverges from the actual label. As we are using Binary Cross-Entropy loss
function, the only activation function compatible is Sigmoid as Softmax function is only to guarantee
that the output is within range.
The results show that an effective method of increasing CNN performance is to use Data Augmentation
in preprocessing, tuning hyperparameters as adding more epochs. Dropout and Batch Normalization
helped to lower overfitting. The final model (Figure 8) performed well with over 99% accuracy.
![Results](https://i.ibb.co/34f57zN/image.png "Results")
![Acc](https://i.ibb.co/cTpSwn1/image.png)
