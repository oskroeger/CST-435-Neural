import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for model input
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

# One-hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Function to yield batches of digit images for training
def batch_digit_stream(images, labels, batch_size=64):
    while True:
        indices = np.random.choice(len(images), batch_size, replace=False)
        yield images[indices], labels[indices]

# Define the CNN model with increased dropout
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.6),  # Increased dropout
    Dense(10, activation='softmax')
])

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)

# Prepare the data stream with a larger batch size
batch_size = 64
stream = batch_digit_stream(train_images, train_labels, batch_size=batch_size)

# Train and evaluate in a loop with validation checks on the test set
for step, (batch_images, batch_labels) in enumerate(stream):
    # Train the model on the batch
    history = model.fit(batch_images, batch_labels, epochs=1, verbose=0, callbacks=[reduce_lr])

    # Predict on the batch and evaluate performance
    predictions = model.predict(batch_images)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(batch_labels, axis=1)

    # Calculate metrics
    batch_accuracy = accuracy_score(actual_labels, predicted_labels)
    loss = history.history['loss'][0]

    # Validate on the test set every 5 batches
    if step % 5 == 0:
        val_loss, val_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Display metrics for the batch
    print(f"Batch {step + 1}:")
    print(f"  Batch Loss: {loss:.4f}")
    print(f"  Batch Accuracy: {batch_accuracy:.4f}")

    # Display one of the digit images with prediction and actual label
    sample_idx = 0
    plt.imshow(batch_images[sample_idx].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[sample_idx]}, Actual: {actual_labels[sample_idx]}")
    plt.axis('off')
    plt.show()

    # Pause and wait for user input to continue
    input("Press Enter to continue to the next batch or Ctrl+C to exit...")
