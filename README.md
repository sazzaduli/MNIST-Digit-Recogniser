# MNIST Neural Network 

This project implements a simple two-layer neural network (no ML libraries!) to classify handwritten digits from the [MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer). 

## MNIST Dataset
MNIST is a collection of handwritten digits from 0 to 9.  
Each image is grayscale and of size **28 x 28 pixels**, centred and size-normalised.

## Requirements
- Python 3.5+
- Scikit-learn (latest version)
- Numpy (+ MKL for Windows)
- Matplotlib

## Introduction
The MNIST dataset contains **70,000** images of handwritten digits:
- **60,000** for training
- **10,000** for testing

These images are already centred and scaled, making it easy to get started quickly with modelling.

## Accuracy
It achieved 83.7% of accuracy on the test set of this CNN model trained on a GPU.

