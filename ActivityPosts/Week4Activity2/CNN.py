from tensorflow.keras import models, layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load and preprocess the data
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
data_train = trainX.reshape(-1, 28, 28, 1) / 255.0
data_test = testX.reshape(-1, 28, 28, 1) / 255.0
labels_train = to_categorical(trainY)
labels_test = to_categorical(testY)

# Function to build the model with modified kernel sizes, strides, and padding
def build_model():
    model = models.Sequential()
    
    # First convolutional layer with modified kernel size, stride, and padding
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Second convolutional layer with different kernel size and padding
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Third convolutional layer to capture deeper features
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build and summarize the model
model = build_model()
model.summary()

# Train the model
model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=10, batch_size=100, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(data_test, labels_test)
print(f"Test Accuracy: {test_acc:.4f}")
