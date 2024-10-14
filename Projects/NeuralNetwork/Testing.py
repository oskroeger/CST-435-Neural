import tensorflow as tf
from tensorflow.keras import layers

# Load the MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3), alpha=0.75)

# Print the summary of the MobileNetV2 base model
base_model.summary()

# Loop through layers and print their configurations
for layer in base_model.layers:
    if isinstance(layer, layers.Conv2D):  # Only interested in Conv2D layers
        print(f"Layer Name: {layer.name}")
        print(f"Filters: {layer.filters}")
        print(f"Kernel Size: {layer.kernel_size}")
        print(f"Padding: {layer.padding}")
        print(f"Activation: {layer.activation}")
        print(f"Input Shape: {layer.input.shape}")  # Access input shape this way
        print("------------------------------------------------")
