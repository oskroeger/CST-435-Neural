import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load a slightly larger fraction of the lm1b dataset for improved accuracy
data, info = tfds.load("lm1b", split='train[:0.2%]', with_info=True, as_supervised=True)
print("Dataset loaded successfully (using 0.2% of data for testing).")

# Convert the dataset into a list of sentences for easier processing
text_data = []
for sentence in data:
    text_data.append(sentence[0].numpy().decode("utf-8"))

# Join sentences into a single text
text_data = " ".join(text_data).lower()

# Initialize the Tokenizer with a limited vocabulary size
vocab_size = 10000  # Limit to the top 10,000 words
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([text_data])

# Prepare sequences from the text data
input_sequences = []
for line in text_data.split("."):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Limit the maximum sequence length
max_sequence_len = min(max([len(x) for x in input_sequences]), 50)  # Set a cap, e.g., 50
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

# Split sequences into features and labels
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, 100, input_length=max_sequence_len-1),
    Masking(mask_value=0.0),
    LSTM(150, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

# Compile the model using Sparse Categorical Crossentropy to avoid one-hot encoding
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model with 4 epochs and a larger data split
checkpoint = ModelCheckpoint("best_lstm_model.keras", monitor="val_loss", save_best_only=True, mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

history = model.fit(X, y, epochs=4, validation_split=0.2, callbacks=[checkpoint, early_stopping], batch_size=64)

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Define text generation function
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = tokenizer.index_word[np.argmax(predicted)]
        seed_text += " " + output_word
    return seed_text

# Generate example text
print(generate_text("The purpose of life is", 5, model, max_sequence_len))

# Evaluate model performance
loss, accuracy = model.evaluate(X, y)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
