from network import X_train, Y_train, gradient_descent, test_prediction

def train():
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.15)

    # example predictions
    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)

train()