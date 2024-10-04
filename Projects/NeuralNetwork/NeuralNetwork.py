import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Load the CSV files
train_df = pd.read_csv('Training_set.csv')  # Ensure correct path
test_df = pd.read_csv('Testing_set.csv')    # Ensure correct path

# Set image directories
train_dir = 'train/'  # Path to folder containing training images
test_dir = 'test/'    # Path to folder containing test images

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizing the image pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for test data, no augmentation

# Training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',   # Column with image filenames
    y_col='label',      # Column with class labels
    target_size=(64, 64),  # Reduced image size to make training faster
    batch_size=16,        # Use a reasonable batch size
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True
)

# Test data generator (without labels)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',   # Column with image filenames
    y_col=None,         # No labels for the test set
    target_size=(64, 64),  # Same reduced image size as training
    batch_size=16,
    class_mode=None,    # No labels, so set to None
    shuffle=False       # Do not shuffle to preserve the order of filenames
)

# Load the pre-trained MobileNetV2 model (with weights pre-trained on ImageNet)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the new model on top of MobileNetV2
model = models.Sequential([
    base_model,  # Add the pre-trained MobileNetV2 base model
    layers.GlobalAveragePooling2D(),  # Pooling layer to reduce dimensionality
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(75, activation='softmax')  # Output layer with 75 classes for butterfly species
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (no validation for simplicity)
history = model.fit(
    train_generator,
    epochs=10
)

# Predict on the test set (without labels)
test_predictions = model.predict(test_generator)

# Convert the predictions (logits) to class labels (indices)
predicted_labels = tf.argmax(test_predictions, axis=1).numpy()  # Convert Tensor to NumPy array

# Convert predicted labels (integers) to their corresponding class names
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
predicted_class_names = [label_map[label] for label in predicted_labels]

# Add the predicted labels to the test DataFrame
test_df['predicted_label'] = predicted_class_names

# Save predictions to a CSV file
test_df.to_csv('submission.csv', columns=['filename', 'predicted_label'], index=False)

print("Predictions saved to submission.csv.")
