# Butterfly Image Classification Neural Network

## About the Dataset
The dataset used in this project features 75 different classes of butterflies. It contains over 1,000 labeled images, including validation images, with each image belonging to only one butterfly category. The dataset includes two CSV files:

- **Training_set.csv**: This file contains the filenames of images and their corresponding labels.
- **Testing_set.csv**: This file contains only the filenames of images in the `test/` folder.

The dataset is too large to upload (even as a zip file) to Halo.
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification?resource=download).

## Steps to Run the Code

### Install Required Libraries
Make sure you have the following Python libraries installed:

- **tensorflow**
- **pandas**
- **pillow**
- **matplotlib**

You can install them using pip:

*pip install tensorflow pandas pillow matplotlib*

### Download and Extract the Dataset
Download the dataset from Kaggle and extract it. Ensure you have the following files and directories after extraction:

- `Training_set.csv`
- `Testing_set.csv`
- `train/` (containing all the training images)
- `test/` (containing the images for testing)

### Organize Dataset Files
Place the dataset files (`Training_set.csv`, `Testing_set.csv`) and the image folders (`train/` and `test/`) in the same directory as the Python script.

### Run the Script
Run the Python script (NeuralNetwork.py) using your preferred method, such as the command line:

*python3 NeuralNetwork.py*

### Results
- The predictions for the test images will be saved in a CSV file called `submission.csv` in the same directory as the script.
- A random sample of four test images, along with their predicted labels, will be displayed using `matplotlib`.

### Early Stopping
The model is set to run for up to 50 epochs, but it will stop early if it achieves high accuracy (around 99%). A message will be displayed, explaining why the training stopped early once the accuracy threshold is reached.

## Detailed Code Breakdown

### Imports
The script begins by importing the necessary libraries: `tensorflow`, `keras`, `pandas`, `PIL`, and `matplotlib`. 

### Loading the Dataset
The training and test CSV files (`Training_set.csv` and `Testing_set.csv`) are loaded using `pandas`. These CSVs contain the filenames of the images and their corresponding labels for the training set. The test set CSV only contains filenames, which need to be classified by the model.

### Data Preprocessing
The images are loaded and preprocessed using the `ImageDataGenerator` class from Keras. This involves resizing the images to 96x96 pixels and normalizing their pixel values by rescaling them to the range [0, 1]. Data augmentation is applied to the training set with horizontal flipping to improve model generalization.

### Model Architecture
A pre-trained MobileNetV2 model is used as the base for transfer learning. This model is loaded without the top classification layer, allowing us to build a new classifier on top. The following layers are added:

- **GlobalAveragePooling2D**: This layer is used to reduce the spatial dimensions after the feature extraction of the base model.
- **Dense Layer (128 units, ReLU activation)**: A fully connected layer that learns representations from the base model’s output.
- **Dense Layer (75 units, Softmax activation)**: The final output layer with 75 units (one for each butterfly class) using softmax for multi-class classification.

### Model Compilation
The model is compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric. Mixed precision training is enabled for faster computation and reduced memory usage.

### Training and Early Stopping
The model is trained using the training data generator. Early stopping is applied, which will stop training once the model's accuracy stalls (around 99%).

### Prediction
After training, the model predicts labels for the test set images. The predicted labels are mapped back to their class names and saved to a `submission.csv` file.

### Displaying Random Predictions
A random sample of four test images is displayed along with their predicted labels. The images are displayed in their original size without resizing.
