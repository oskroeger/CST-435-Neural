import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("mnist_sequence_model.h5")

# Preprocess each digit for the 4-digit image
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

# Preprocess the whole 4-digit image
def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Assume the digits are aligned horizontally; split them
    # Adjust these values based on your image
    digit_width = img.shape[1] // 4  # assuming 4 digits evenly spaced
    digits = [img[:, i * digit_width:(i + 1) * digit_width] for i in range(4)]
    
    # Preprocess each digit
    processed_digits = [preprocess_digit(digit) for digit in digits]
    
    # Stack into a single sequence batch for the model
    return np.array(processed_digits).reshape(1, 4, 28, 28, 1)

# Path to your handwritten 4-digit image
image_path = 'path_to_your_4_digit_image.jpg'

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
