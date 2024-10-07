import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Enable mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ---------------------------------------
# Load and prepare the dataset
# ---------------------------------------
# Load the CSV files containing image filenames and labels
train_df = pd.read_csv('Training_set.csv')
test_df = pd.read_csv('Testing_set.csv')

# Set the directories containing the images
train_dir = 'train/'
test_dir = 'test/'

# ---------------------------------------
# Preprocess images for training and testing
# ---------------------------------------
# Data augmentation and rescaling for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale images to [0, 1] range
    horizontal_flip=True       # Simple augmentation by flipping images horizontally
)

# Rescale the test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate training data with reduced image size (96x96) and batch size of 32
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=(96, 96),      # Reduced image size for faster training
    batch_size=32,
    class_mode='categorical',  # Categorical labels for multi-class classification
    shuffle=True
)

# Generate test data (without labels) for prediction
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col=None,                # No labels needed for the test set
    target_size=(96, 96),      # Match training image size
    batch_size=32,
    class_mode=None,
    shuffle=False              # Don't shuffle to maintain order in prediction
)

# ---------------------------------------
# Initialize the pre-trained MobileNetV2 CNN
# ---------------------------------------
# MobileNetV2 model, pretrained on ImageNet, used as the base
# The first layer (Conv1) has the following arguments:
# - Filters: 24
# - Kernel size: (3, 3)
# - Padding: 'same'
# - Activation: ReLU
# - Input shape: (96, 96, 3)
# Global Average Pooling is applied as the pooling operation
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3), alpha=0.75)

# Freeze the base model's convolutional layers
base_model.trainable = False  # Freeze to only train the top layers for faster training

# ---------------------------------------
# Build the complete model
# ---------------------------------------
# Add the base model, global pooling, and fully connected layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),    # Global pooling to down-sample features
    layers.Dense(128, activation='relu'),  # Fully connected layer (ReLU activation)
    layers.Dense(75, activation='softmax')  # Output layer (Softmax for multi-class classification)
])

# ---------------------------------------
# Compile the model
# ---------------------------------------
# Compile using categorical crossentropy, Adam optimizer, and accuracy as the metric
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------
# Train the model with early stopping
# ---------------------------------------
# Early stopping will monitor accuracy, and stop training after 2 epochs of no improvement
early_stopping = EarlyStopping(monitor='accuracy', patience=2, restore_best_weights=True)

# Train the model for up to 50 epochs but stop early if accuracy is high enough
history = model.fit(
    train_generator,
    epochs=50,  # Train up to 50 epochs
    callbacks=[early_stopping],  # Stop early if accuracy plateaus
    verbose=1
)

if len(history.history['accuracy']) < 50:
    print(f"Training stopped early after {len(history.history['accuracy'])} epochs because accuracy plateaued.")

# ---------------------------------------
# Predict on the test set
# ---------------------------------------
# Predict the class labels for the test set
test_predictions = model.predict(test_generator)
predicted_labels = tf.argmax(test_predictions, axis=1).numpy()  # Get the class index with the highest score

# Map predicted class indices back to class names
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
predicted_class_names = [label_map[label] for label in predicted_labels]

# Save predictions to a CSV file
test_df['predicted_label'] = predicted_class_names
test_df.to_csv('submission.csv', columns=['filename', 'predicted_label'], index=False)

print("Predictions saved to submission.csv.")

# ---------------------------------------
# Visualize Random Test Predictions
# ---------------------------------------
# Function to display 4 random test images along with their predicted labels
import random
def display_random_predictions(test_df, test_dir, predicted_class_names):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # Create a 2x2 grid for the images
    axes = axes.ravel()  # Flatten the axes array for easier indexing

    # Select 4 random images from the test dataset
    random_indices = random.sample(range(len(test_df)), 4)

    for i, idx in enumerate(random_indices):
        img_path = os.path.join(test_dir, test_df['filename'].iloc[idx])  # Get the image path
        
        # Load the original image (without resizing)
        original_img = Image.open(img_path)
        
        # Display the original image in the grid
        axes[i].imshow(original_img)
        axes[i].axis('off')  # Hide axis for a cleaner display
        axes[i].set_title(f"Pred: {predicted_class_names[idx]}")  # Show predicted class

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

# Display 4 random test images and their predictions
display_random_predictions(test_df, test_dir, predicted_class_names)

# Plot accuracy and loss over epochs
def plot_accuracy_and_loss(history):
    # Get the values from the history object
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(accuracy) + 1)
    
    # Plot accuracy
    plt.figure(figsize=(14, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro-', label='Training loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Call the function after training
plot_accuracy_and_loss(history)
