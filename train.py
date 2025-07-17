from network import labels, features, gradient_descent, test_prediction

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, axis=0)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = features[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = labels[index]
    print("Prediction:", prediction)
    print("Label:", label)
    
    # Reshape for visualization (28x28) and scale pixel values
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(34, W1, b1, W2, b2)
test_prediction(14, W1, b1, W2, b2)
test_prediction(22, W1, b1, W2, b2)
test_prediction(42, W1, b1, W2, b2)

# Calculate accuracy on training data as a sanity check
train_preds = predict(W1, b1, W2, b2, features)
print(train_preds)
train_accuracy = np.mean(train_preds == labels)
print(f"Training accuracy: {train_accuracy}")
