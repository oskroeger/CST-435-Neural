import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)  # One-hot encode

# Define a simple neural network without regularization
def create_model(with_regularization=False):
    if with_regularization:
        regularizer = l2(0.001)  # L2 regularization with lambda = 0.001
    else:
        regularizer = None

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dense(64, activation='relu', kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    return model

# Compile and train the model without regularization
model_no_reg = create_model(with_regularization=False)
model_no_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_no_reg = model_no_reg.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Compile and train the model with L2 regularization
model_with_reg = create_model(with_regularization=True)
model_with_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_with_reg = model_with_reg.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate both models on the test set
test_loss_no_reg, test_acc_no_reg = model_no_reg.evaluate(x_test, y_test, verbose=0)
test_loss_with_reg, test_acc_with_reg = model_with_reg.evaluate(x_test, y_test, verbose=0)

print(f"Test accuracy without regularization: {test_acc_no_reg:.4f}")
print(f"Test accuracy with L2 regularization: {test_acc_with_reg:.4f}")

# Plot training and validation accuracy and loss
plt.figure(figsize=(14, 5))

# Accuracy plots
plt.subplot(1, 2, 1)
plt.plot(history_no_reg.history['accuracy'], label='Train No Reg')
plt.plot(history_no_reg.history['val_accuracy'], label='Val No Reg')
plt.plot(history_with_reg.history['accuracy'], label='Train L2 Reg')
plt.plot(history_with_reg.history['val_accuracy'], label='Val L2 Reg')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plots
plt.subplot(1, 2, 2)
plt.plot(history_no_reg.history['loss'], label='Train No Reg')
plt.plot(history_no_reg.history['val_loss'], label='Val No Reg')
plt.plot(history_with_reg.history['loss'], label='Train L2 Reg')
plt.plot(history_with_reg.history['val_loss'], label='Val L2 Reg')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
