# consumer.py
import websocket
import pandas as pd
import matplotlib.pyplot as plt
import json
from model import build_ann_model, preprocess_data, fit_scaler
import numpy as np

# Initialize an empty DataFrame to collect transaction data
df = pd.DataFrame(columns=["transactionID", "userID", "amount", "timestamp", "itemID", "fraud_prediction"])

# Initialize a DataFrame to store fraudulent transactions
fraudulent_transactions = pd.DataFrame(columns=["transactionID", "userID", "amount", "timestamp", "itemID", "fraud_prediction"])

# Improved synthetic training data
initial_training_data = np.array([
    [1, 500.0, 1],   # Fraud example
    [2, 300.0, 2],
    [3, 150.0, 3],
    [4, 50.0, 4],
    [5, 20.0, 5],
    [6, 1000.0, 6],
    [7, 60.0, 7],
    [8, 70.0, 8],
    [9, 700.0, 9],
    [10, 80.0, 10],
    [11, 20.0, 11],
    [12, 15.0, 12],
    [13, 400.0, 13],
    [14, 200.0, 14],
    [15, 100.0, 15],
    [16, 800.0, 16],
    [17, 10.0, 17],
    [18, 250.0, 18],
])
fit_scaler(initial_training_data)  # Fit the scaler using initial data

# Initialize the ANN model
model = build_ann_model(input_shape=3)  # Adjust input_shape based on the number of features

# Train with the improved synthetic dataset
y_initial = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # Labels matching initial data
model.fit(preprocess_data(initial_training_data), y_initial, epochs=50, verbose=1)  # Train with better data

# Set up the plot
plt.ion()  # Enable interactive mode for real-time updates
fig, ax = plt.subplots()
plt.xlabel("Transaction Count")
plt.ylabel("Fraud Prediction Score")
plt.title("Real-Time Fraud Prediction")
plt.show()  # Show the plot window

# Define function to update plot data
def update_plot(df):
    ax.clear()  # Clear the current plot
    ax.plot(df.index, df["fraud_prediction"], 'r-')  # Plot the fraud prediction scores
    ax.set_xlabel("Transaction Count")
    ax.set_ylabel("Fraud Prediction Score")
    ax.set_title("Real-Time Fraud Prediction")
    plt.pause(0.05)  # Pause briefly to update the plot

# Define function to display fraudulent transactions
def display_fraudulent_transactions(transaction):
    global fraudulent_transactions
    # Append the fraudulent transaction to the DataFrame
    fraudulent_transactions = pd.concat([fraudulent_transactions, transaction], ignore_index=True)
    # Print the table of fraudulent transactions
    print("\n=== Fraudulent Transactions ===")
    print(fraudulent_transactions.to_string(index=False))  # Display without index numbers

# WebSocket message handling
def on_message(ws, message):
    global df
    print(f"Received: {message}")
    
    # Convert message to DataFrame
    transaction = pd.json_normalize(json.loads(message))
    transaction["fraud_prediction"] = 0.0  # Initialize the prediction column

    # Extract features and preprocess for ANN input
    features = transaction[["userID", "amount", "transactionID"]].values.astype(float)
    features_preprocessed = preprocess_data(features)

    # Make fraud prediction using the ANN model
    prediction = model.predict(features_preprocessed)
    transaction["fraud_prediction"] = prediction[0][0]  # Save prediction score

    # Append the new transaction data to the DataFrame
    df = pd.concat([df, transaction], ignore_index=True)

    # Display the distribution of fraud prediction scores periodically
    if len(df) % 10 == 0:
        print("\n=== Fraud Prediction Score Distribution ===")
        print(df["fraud_prediction"].describe())  # Display stats to help adjust threshold

    # Check if the transaction is predicted as fraudulent
    fraud_threshold = 0.3  # Adjust this threshold based on score distribution inspection
    if transaction["fraud_prediction"].iloc[0] > fraud_threshold:  # Adjusted threshold for fraud detection
        display_fraudulent_transactions(transaction)

    # Update the plot with the latest data
    update_plot(df)

def on_open(ws):
    print("Connection opened.")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed.")

if __name__ == "__main__":
    # Start WebSocket connection to the producer
    ws = websocket.WebSocketApp("ws://127.0.0.1:9999/",
                                on_message=on_message,
                                on_open=on_open,
                                on_close=on_close)
    ws.run_forever()  # Keep the WebSocket connection alive
