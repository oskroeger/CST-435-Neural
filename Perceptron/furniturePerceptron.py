import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Sigmoid activation function for training
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary activation function for final predictions
def activation_function(x):
    return 1 if x >= 0 else 0

# Perceptron training function with intermediate heatmap captures and error tracking
def train_perceptron(data, labels, epochs=200, learning_rate=0.2):
    # Initialize weights and bias with small random values for more nuanced updates
    weights = np.random.randn(data.shape[1]) * 0.01
    bias = 0.0
    intermediate_weighted_sums = []  # To store intermediate weighted sums
    errors = []  # To track error over epochs

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(data)):
            x = data[i]
            y = labels[i]

            # Calculate the weighted sum and use sigmoid for training
            weighted_sum = np.dot(weights, x) + bias
            prediction = sigmoid(weighted_sum)  # Use sigmoid for more varied weight updates

            # Calculate the gradient with respect to the sigmoid output
            error = y - prediction
            total_error += error ** 2  # Accumulate squared error
            weights += learning_rate * error * x * prediction * (1 - prediction)  # Derivative of sigmoid
            bias += learning_rate * error * prediction * (1 - prediction)

        # Store total error for the epoch
        errors.append(total_error)

        # Capture weighted sums at the start, halfway, and at the end
        if epoch == 0 or epoch == epochs // 2 or epoch == epochs - 1:
            current_weighted_sums = [np.dot(weights, data[i]) + bias for i in range(len(data))]
            intermediate_weighted_sums.append(np.array(current_weighted_sums).reshape(10, 10))

    return weights, bias, intermediate_weighted_sums, errors

# Manually inputting the grid values into training_data and corresponding labels
training_data = np.eye(100)  # 100 vectors, each one representing a grid cell (one-hot encoded)

# Labels manually inputted based on the grid data (flattened from the grid)
labels = [
    1, 1, 0, 0, 1, 1, 0, 0, 1, 1,  # Row 1
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,  # Row 2
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Row 3
    0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  # Row 4
    1, 0, 0, 1, 1, 1, 1, 0, 0, 1,  # Row 5
    1, 0, 0, 1, 1, 1, 1, 0, 0, 1,  # Row 6
    0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  # Row 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Row 8
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1,  # Row 9
    1, 1, 0, 0, 1, 1, 0, 0, 1, 1  # Row 10
]

# Train the perceptron using the sigmoid function for training and capture intermediate states and errors
weights, bias, intermediate_weighted_sums, errors = train_perceptron(training_data, labels, epochs=200,
                                                                     learning_rate=0.2)

# Final testing to capture the final predictions
weighted_sums = [np.dot(weights, training_data[i]) + bias for i in range(100)]
binary_predictions = [activation_function(ws) for ws in weighted_sums]

# Reshape weighted sums, binary predictions, and labels to 10x10 grids for visualization
grid_weighted_sums = np.array(weighted_sums).reshape(10, 10)
grid_predictions = np.array(binary_predictions).reshape(10, 10)
grid_expected = np.array(labels).reshape(10, 10)

# Use symmetric normalization to enhance contrast evenly for both reds and blues
norm = Normalize(vmin=-np.max(np.abs(grid_weighted_sums)), vmax=np.max(np.abs(grid_weighted_sums)))

# Define the figure layout using GridSpec for precise control
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.3)  # Define a 2x4 grid with adjusted spacing

# Plotting the heatmaps in the first row
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(intermediate_weighted_sums[0], cmap='RdBu_r', interpolation='nearest', norm=norm)
ax1.set_title('Heatmap at Start of Training')
ax1.set_xlabel('Columns')
ax1.set_ylabel('Rows')

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(intermediate_weighted_sums[1], cmap='RdBu_r', interpolation='nearest', norm=norm)
ax2.set_title('Heatmap Midway Through Training')
ax2.set_xlabel('Columns')
ax2.set_ylabel('Rows')

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(intermediate_weighted_sums[2], cmap='RdBu_r', interpolation='nearest', norm=norm)
ax3.set_title('Final Heatmap of Weighted Sums')
ax3.set_xlabel('Columns')
ax3.set_ylabel('Rows')

# Create a dedicated axis for the colorbar on the right side of the grid
cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(im3, cax=cbar_ax, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label('Weighted Sum (Gradient of Optimality)')

# Plotting the final binary decisions and expected values, and the error plot
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(grid_predictions, cmap='gray', interpolation='nearest')
ax4.set_title('Final Binary Decisions')
ax4.set_xlabel('Columns')
ax4.set_ylabel('Rows')

ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(grid_expected, cmap='gray', interpolation='nearest')
ax5.set_title('Expected Values')
ax5.set_xlabel('Columns')
ax5.set_ylabel('Rows')

ax6 = fig.add_subplot(gs[1, 2:])
ax6.plot(errors, label='Training Error')
ax6.set_title('Error Reduction Over Epochs')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Total Squared Error')
ax6.legend()

# Manually adjust layout to ensure all plots are square and evenly spaced
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

plt.show()

