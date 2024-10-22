import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Step 1: Load the raw text from file
with open('alicetext.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Optionally use the full text instead of a subset
# Use a subset of the text (e.g., the first 100,000 characters) for faster testing, or comment this line to use full text
text = text[:100000]

# Step 2: Tokenize the characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1

# Convert text to sequence of integers
sequences = tokenizer.texts_to_sequences([text])[0]

# Define shorter sequence length (e.g., 20 characters per sequence)
seq_length = 20

# Create input-output pairs for training
inputs = []
outputs = []
for i in range(0, len(sequences) - seq_length):
    inputs.append(sequences[i:i + seq_length])
    outputs.append(sequences[i + seq_length])

inputs = np.array(inputs)
outputs = np.array(outputs)

# Step 3: Build the RNN model with more LSTM units and layers
model = Sequential()
model.add(Embedding(total_chars, 50, input_length=seq_length))
model.add(LSTM(128, return_sequences=True))  # Increased LSTM units
model.add(LSTM(128))  # Second LSTM layer with increased units
model.add(Dense(total_chars, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Step 4: Train the model with increased batch size and more epochs
model.fit(inputs, outputs, epochs=30, batch_size=256)

# Step 5: Helper function to sample predictions with temperature control
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Step 6: Generate text function with temperature control
def generate_text(seed_text, num_chars, temperature=1.0):
    # Truncate the seed_text if it is longer than seq_length
    if len(seed_text) > seq_length:
        seed_text = seed_text[-seq_length:]

    for _ in range(num_chars):
        # Tokenize the seed text into integer sequence
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Ensure the length of tokenized_input is not greater than seq_length
        tokenized_input = np.pad(tokenized_input, (seq_length-len(tokenized_input), 0), 'constant')
        
        # Reshape for model input
        tokenized_input = np.reshape(tokenized_input, (1, seq_length))

        # Predict next character
        predictions = model.predict(tokenized_input)[0]
        predicted_char_index = sample(predictions, temperature)

        # Decode predicted index back to character
        predicted_char = tokenizer.sequences_to_texts([[predicted_char_index]])[0]

        # Append the predicted character to the seed text
        seed_text += predicted_char
        
        # Ensure the seed_text is truncated to the last seq_length characters
        seed_text = seed_text[-seq_length:]

    return seed_text

# Example of text generation
seed_text = "alice was beginning to get very"
generated_text = generate_text(seed_text, 200, temperature=0.5)
print(generated_text)
