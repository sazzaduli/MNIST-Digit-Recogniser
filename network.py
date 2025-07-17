import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 

# Load training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Convert to numpy arrays
train_data = np.array(train_df)
test_data = np.array(test_df)

# testing and training data
np.random.shuffle(train_data) 

# Transpose for easier slicing
train_data = train_data.T
test_data = test_data.T

labels = train_data[0].astype(int) 
features = train_data[1:] / 255.0
test_data = test_data / 255.0


# Define Network Functions
def init_parameters():
    '''
    Randomly generates the weights and biases for the first generation neural network.
    '''

    W1 = np.random.rand(10, 784) - 0.5  
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5   
    b2 = np.random.rand(10, 1) - 0.5    

    return W1, b1, W2, b2


def relu(Z):
    '''''
    ReLU activation function.
    '''
    
    return np.maximum(0, Z)

def softmax(Z):
    '''
    Softmax activation function for the output layer.
    '''
    
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(W1, b1, W2, b2, X):
    '''
    Perform a forward pass through the network.
    X -- represents the input data/input layer
    '''

    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


def one_hot_encode(Y):
    '''
    Formats the expected value into an array of the correct size and format.
    '''

    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1
    
    return one_hot.T

def relu_derivative(Z):
    '''
    Derivative of ReLU activation.
    '''

    return Z > 0

def back_propagation(Z1, A1, Z2, A2, W2, X, Y):
    '''
    Runs the data from the output nodes to the input nodes and
    Calculates the error in a backwards motion.
    '''

    m = Y.size
    one_hot_Y = one_hot_encode(Y)

    dZ2 = 2 * (A2 - one_hot_Y)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    '''
    Update weights and biases using gradients and learning rate.
    '''

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

def gradient_descent(X, Y, learning_rate=0.15, print_every=10, accuracy_threshold=0.999, max_iterations=500):
    """
    Train the neural network until the accuracy threshold is met or the maximum number of iterations is reached.
    
    Parameters:
        X: input features
        Y: labels
        learning_rate: step size for parameter updates
        print_every: iterations interval for printing accuracy
        accuracy_threshold: stopping accuracy criterion
        max_iterations: hard stop for training loop
    
    Returns:
        Trained parameters W1, b1, W2, b2
    """
    W1, b1, W2, b2 = init_parameters()
    iteration = 0

    while iteration <= max_iterations:
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if iteration % print_every == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = np.mean(predictions == Y)
            print(f"Iteration: {iteration}")
            print(predictions, Y)
            print(f"Accuracy: {accuracy:.6f}")

            if accuracy >= accuracy_threshold:
                print("Reached desired accuracy!")
                break

        iteration += 1

    return W1, b1, W2, b2

def predict(W1, b1, W2, b2, X):
    """ Make predictions for input data X using trained parameters."""
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    return np.argmax(A2, axis=0)


def create_submission(predictions):
    """ Create submission CSV from predictions."""
    submission_df = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions
    })
    submission_df.to_csv("submission.csv", index=False)
    
