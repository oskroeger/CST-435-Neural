import numpy as np

# Example MLP forward propagation with ReLU activation and a threshold output
def relu(z):
    return np.maximum(0, z)

def threshold_function(z, threshold=0.5):
    return 1 if z >= threshold else 0

# Input data
x = np.array([1.0, 2.0])  # 2D input

# Weights and biases for a simple MLP with one hidden layer and one output
w_hidden = np.array([[0.5, -0.2], [0.3, 0.8]])  # Weights for hidden layer
b_hidden = np.array([0.0, 0.0])  # Bias for hidden layer
w_output = np.array([0.7, -0.5])  # Weights for output layer
b_output = 0.1  # Bias for output layer

# Forward propagation
# Hidden layer computation
hidden_input = np.dot(x, w_hidden) + b_hidden
hidden_output = relu(hidden_input)

# Output layer computation
output = np.dot(hidden_output, w_output) + b_output

# Applying threshold function to the final output
final_output = threshold_function(output, threshold=0.5)

print("Predicted Output (after threshold):", final_output)