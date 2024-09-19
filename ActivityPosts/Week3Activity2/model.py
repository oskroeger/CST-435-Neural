# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to build the ANN model
def build_ann_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    
    # Compile the model with binary cross-entropy loss and adam optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess data for the ANN model
scaler = StandardScaler()

def preprocess_data(data):
    return scaler.transform(data)

def fit_scaler(data):
    scaler.fit(data)
