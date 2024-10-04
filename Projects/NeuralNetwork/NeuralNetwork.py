import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Enable mixed precision training (if supported)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Load the CSV files
train_df = pd.read_csv('Training_set.csv')
test_df = pd.read_csv('Testing_set.csv')

# Set image directories
train_dir = 'train/'
test_dir = 'test/'

# Simplified Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True  # Simple augmentation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator with reduced image size and increased batch size
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=(96, 96),  # Reduced image size to make training faster
    batch_size=32,         # Increased batch size
    class_mode='categorical',
    shuffle=True
)

# Test data generator (without labels)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col=None,
    target_size=(96, 96),  # Match image size with training set
    batch_size=32,         # Increased batch size for efficiency
    class_mode=None,
    shuffle=False
)

# Load a smaller MobileNetV2 model with alpha=0.75 for fewer parameters
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3), alpha=0.75)

# Freeze the base model layers initially
base_model.trainable = False

# Build the new model on top of MobileNetV2
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(75, activation='softmax')  # Output layer with 75 classes for butterfly species
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='accuracy', patience=2, restore_best_weights=True)

# Train the model (stop after 10 epochs as accuracy is already high)
history = model.fit(
    train_generator,
    epochs=10,  # Stop after the first wave of 10 epochs
    callbacks=[early_stopping],
    verbose=1
)

# Predict on the test set (without labels)
test_predictions = model.predict(test_generator)
predicted_labels = tf.argmax(test_predictions, axis=1).numpy()

# Convert predicted labels to class names
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
predicted_class_names = [label_map[label] for label in predicted_labels]

# Save predictions to a CSV file
test_df['predicted_label'] = predicted_class_names
test_df.to_csv('submission.csv', columns=['filename', 'predicted_label'], index=False)

print("Predictions saved to submission.csv.")

import random

# Function to display 4 random test images along with their predicted labels (in original size)
def display_random_predictions(test_df, test_dir, predicted_class_names):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2x2 grid for images
    axes = axes.ravel()  # Flatten the axes array for easier indexing

    # Select 4 random indices from the test DataFrame
    random_indices = random.sample(range(len(test_df)), 4)

    for i, idx in enumerate(random_indices):  # Loop through the random indices
        img_path = os.path.join(test_dir, test_df['filename'].iloc[idx])  # Get image path
        
        # Load the original image without resizing
        original_img = Image.open(img_path)  # Open the image in its original size
        
        # Display the original image in the grid
        axes[i].imshow(original_img)
        axes[i].axis('off')  # Hide axes
        axes[i].set_title(f"Pred: {predicted_class_names[idx]}")  # Display predicted label

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

# Call the function to display 4 random images and their predictions
display_random_predictions(test_df, test_dir, predicted_class_names)