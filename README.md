# MNIST Neural Network 
This project is all about teaching a neural network how to recognise handwritten digits using the well-known MNIST dataset, implementing a simple two-layer neural network to classify handwritten digits from the [MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer)

The model is a 3-layer feedforward neural network, and it’s trained using backpropagation and gradient descent. Here's a diagram that visualises the network structure I used:
<p align="center"> <img src="https://github.com/sazzaduli/MNIST-Digit-Recogniser/blob/main/res/iHDtO.png"> </p>

## MNIST Dataset
MNIST is a collection of handwritten digits from 0 to 9.  
Each image is grayscale and of size **28 x 28 pixels**, centred and size-normalised.

## Requirements
- Python 3.5+
- Scikit-learn (latest version)
- Numpy (+ MKL for Windows)
- Matplotlib

## Features Implemented
- Forward Propagation
- Backwards Propagation
- Gradient Descent optimisation
- Model fitting (training)
- Predictions on new data
- Accuracy evaluation

## Getting Started
Want to try it yourself? Here’s how to get everything set up:

1. Click on Fork.
2. Clone your fork locally
3. git clone https://github.com/sazzaduli/MNIST-Digit-Recogniser/tree/main
4. pip3 install -r requirements.txt
   
## Accuracy
It achieved 88.6% accuracy on the test set of this CNN model.

