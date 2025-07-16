import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# retrieve testing and training data
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')


# testing data
test_data = np.array(test_data)

test_data = test_data.T # transpose the data
X_test = test_data
X_test = X_test / 255

# training data
train_data = np.array(train_data)
np.random.shuffle(train_data) # shuffle the data around

train_data = train_data.T # transpose the data
Y_train = train_data[0]
X_train = train_data[1:]
X_train = X_train / 255


# initialize weights and biases
def init_params():
    '''
    Randomly generates the weights and biases for the first generation neural network.
    '''

    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def ReLU(Z):
    '''
    Pythonic ReLU function.
    '''

    return np.maximum(0, Z)

def softmax(Z):
    '''
    Pythonic softmax activation function.
    '''

    return np.exp(Z) / sum(np.exp(Z))

# forward propagation
def forward_prop(W1, b1, W2, b2, X):
    '''
    Runs the data from the input nodes to the output nodes in a 
    forward motion.

    X -- represents the input data / input layer
    '''

    z1 = W1.dot(X) + b1
    A1 = ReLU(z1)
    z2 = W2.dot(A1) + b2
    A2 = softmax(z2)

    return z1, A1, z2, A2


def one_hot(Y):
    '''
    Formats the expected value into an array of the correct size and format.
    '''

    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

def deriv_ReLU(Z):
    '''
    Pythonic derivative of ReLU function.
    '''

    return Z > 0

# backward propagation
def back_prop(z1, A1, z2, A2, W1, W2, X, Y):
    '''
    Runs the data from the output nodes to the input nodes and
    calculates the error in a backward motion.
    '''

    m = Y.size
    one_hot_Y = one_hot(Y)

    dz2 = 2 * (A2 - one_hot_Y)
    dW2 = 1 / m * dz2.dot(A1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = W2.T.dot(dz2) * deriv_ReLU(z1)
    dW1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1)

    return dW1, db1, dW2, db2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)

    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    '''
    Adjusts the weights and biases after front and back propagation is
    completed.
    '''

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

# gradient descent
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        z1, A1, z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, A1, z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print('Iteration: ', i)
            print('Accuracy: ', get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2