import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Data Augmentation setup for individual digit images
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Function to create 4-digit sequences from single digit images
def create_sequence_data(images, labels, num_sequences=2000):  # Reduced dataset size
    sequences = []
    sequence_labels = []
    for _ in range(num_sequences):
        indices = np.random.choice(len(images), 4, replace=False)
        seq_images = images[indices]
        seq_labels = labels[indices]
        sequences.append(seq_images)
        sequence_labels.append(seq_labels)
    return np.array(sequences), np.array(sequence_labels)

# Create smaller training and test data for sequences
train_sequences, train_seq_labels = create_sequence_data(train_images, train_labels, num_sequences=2000)
test_sequences, test_seq_labels = create_sequence_data(test_images, test_labels, num_sequences=500)

# One-hot encode sequence labels
train_seq_labels = to_categorical(train_seq_labels, num_classes=10)
test_seq_labels = to_categorical(test_seq_labels, num_classes=10)

# Reshape data for the model
train_sequences = train_sequences.reshape(-1, 4, 28, 28, 1).astype('float32')
test_sequences = test_sequences.reshape(-1, 4, 28, 28, 1).astype('float32')

# Function to apply augmentation to individual digits and reassemble sequences
def augment_sequences(sequences, labels):
    augmented_sequences = []
    augmented_labels = []
    for seq, lbl in zip(sequences, labels):
        augmented_seq = [datagen.random_transform(img.reshape(28, 28, 1)) for img in seq]
        augmented_sequences.append(augmented_seq)
        augmented_labels.append(lbl)
    return np.array(augmented_sequences), np.array(augmented_labels)

# Apply augmentation to training sequences
augmented_train_sequences, augmented_train_labels = augment_sequences(train_sequences, train_seq_labels)

# Define a simplified CNN-LSTM model for recognizing sequences
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(4, 28, 28, 1)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(Flatten()),
    LSTM(128, return_sequences=True),  # Simplified model with fewer units
    Dropout(0.5),
    TimeDistributed(Dense(64, activation='relu')),  # Fewer dense units
    TimeDistributed(Dense(10, activation='softmax'))
])

# Compile the simplified model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train the simplified model
model.fit(
    augmented_train_sequences, augmented_train_labels,
    epochs=20,  # Reduced number of epochs
    batch_size=32,
    validation_data=(test_sequences, test_seq_labels),
    callbacks=[early_stopping, lr_reduction]
)

# Evaluate the simplified model
test_loss, test_acc = model.evaluate(test_sequences, test_seq_labels)
print(f"Test accuracy: {test_acc}")

# Save the simplified model
model.save("mnist_sequence_model_simplified.h5")

# Load the trained model for testing
model = load_model("mnist_sequence_model_simplified.h5")

# Function to preprocess individual digit images
def preprocess_digit(digit_img):
    # Resize digit to 28x28
    digit_img = cv2.resize(digit_img, (28, 28))
    
    # Invert colors if necessary (make the digit white on black)
    if np.mean(digit_img) > 127:  # If background is white, invert
        digit_img = 255 - digit_img

    # Normalize pixel values
    digit_img = digit_img / 255.0
    
    # Reshape to add batch and channel dimensions
    digit_img = digit_img.reshape(28, 28, 1)
    
    return digit_img

# Function to preprocess a 4-digit image into segments
def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Assume the digits are aligned horizontally; split them
    digit_width = img.shape[1] // 4  # assuming 4 digits evenly spaced
    digits = [img[:, i * digit_width:(i + 1) * digit_width] for i in range(4)]
    
    # Preprocess each digit
    processed_digits = [preprocess_digit(digit) for digit in digits]
    
    # Stack into a single sequence batch for the model
    return np.array(processed_digits).reshape(1, 4, 28, 28, 1)

# Path to your handwritten 4-digit image
image_path = 'DIGITS.JPG'

# Preprocess the image
input_sequence = preprocess_image(image_path)

# Predict using the model
predictions = model.predict(input_sequence)

# Convert predictions to readable digits
predicted_labels = [np.argmax(pred) for pred in predictions[0]]

# Print the results
print(f"Predicted 4-digit sequence: {''.join(map(str, predicted_labels))}")

# Visualize the result
plt.figure(figsize=(10, 2))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(input_sequence[0, i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.show()
