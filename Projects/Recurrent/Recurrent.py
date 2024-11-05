# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load a slightly larger dataset for better text diversity
try:
    print("Attempting to load an expanded lm1b dataset...")
    data, info = tfds.load("lm1b", split='train[:0.1%]', with_info=True, as_supervised=True)
    text_data = []
    for sentence in data:
        text_data.append(sentence[0].numpy().decode("utf-8"))
    text_data = " ".join(text_data).lower()  # Convert text to lowercase for uniformity
    print("Loaded a larger subset of the lm1b dataset.")
except:
    # Fall back to Tiny Shakespeare if lm1b is unavailable
    print("lm1b dataset loading failed. Using Tiny Shakespeare dataset instead...")
    data, info = tfds.load("tiny_shakespeare", split='train', with_info=True)
    text_data = ""
    for sentence in data:
        text_data += sentence['text'].numpy().decode("utf-8")
    text_data = text_data.lower()  # Convert text to lowercase for uniformity
    print("Loaded the Tiny Shakespeare dataset.")

# Preprocess text by removing punctuation and ensuring all text is lowercase
print("Preprocessing text data...")
text_data = re.sub(r'[^\w\s]', '', text_data)  # Remove punctuation for simpler tokenization
print(f"Sample preprocessed text: {text_data[:200]}")

# Initialize Tokenizer with an increased vocabulary size to capture frequent words
vocab_size = 5000  # Increased vocabulary size for better representation
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([text_data])

# Filter the tokenizer word index to only keep the top vocab_size words
word_index = {word: index for word, index in tokenizer.word_index.items() if index < vocab_size}
tokenizer.word_index = word_index

# Check tokenizer details to ensure proper vocabulary setup
print(f"Vocabulary size (filtered): {len(tokenizer.word_index)}")
print(f"Sample word-to-integer mapping: {list(tokenizer.word_index.items())[:10]}")

# Prepare sequences for training with a custom data generator
class TextDataGenerator(Sequence):
    def __init__(self, text, tokenizer, max_sequence_len, batch_size=64):
        self.text = text
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.batch_size = batch_size
        self.indexes = np.arange(len(text))
        
        # Create sequences of words for next-word prediction
        token_list = tokenizer.texts_to_sequences([text])[0]
        self.input_sequences = []
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[max(0, i - self.max_sequence_len):i+1]
            self.input_sequences.append(n_gram_sequence)
        
        # Limit the number of sequences for manageable training size
        max_sequences = 100000
        self.input_sequences = self.input_sequences[:max_sequences]
        print(f"Total number of sequences created: {len(self.input_sequences)}")
        
    def __len__(self):
        return len(self.input_sequences) // self.batch_size
    
    def __getitem__(self, idx):
        # Get a batch of sequences and pad them for uniformity
        batch_sequences = self.input_sequences[idx * self.batch_size: (idx + 1) * self.batch_size]
        padded_sequences = pad_sequences(batch_sequences, maxlen=self.max_sequence_len, padding='pre')
        
        X = padded_sequences[:, :-1]  # Inputs
        y = padded_sequences[:, -1]   # Labels (next word)
        return X, y

# Set sequence length for better context and initialize data generator
max_sequence_len = 15  # Increased sequence length for more context
batch_size = 64  # Batch size for efficient training
data_generator = TextDataGenerator(text_data, tokenizer, max_sequence_len, batch_size)

# Load GloVe embeddings for pretrained word vectors
print("Loading GloVe embeddings...")
embeddings_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print(f"Total words in GloVe embeddings: {len(embeddings_index)}")

# Prepare embedding matrix, using pretrained GloVe vectors for words in our vocabulary
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Display embedding matrix details for verification
print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Sample embedding for 'king' (if in vocabulary): {embedding_matrix[tokenizer.word_index.get('king', 0)]}")

# Build the LSTM model as specified, with an additional LSTM layer for complexity
print("Building the model...")
model = Sequential([
    # Embedding layer with pretrained GloVe weights (non-trainable)
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_shape=(max_sequence_len-1,), trainable=False),
    Masking(mask_value=0.0),  # Mask padding values
    LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # First LSTM layer
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Second LSTM layer for added complexity
    Dense(vocab_size, activation='softmax')  # Output layer with softmax for next-word prediction
])

# Compile model with Adam optimizer and track accuracy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model with checkpoints and early stopping to prevent overfitting
checkpoint = ModelCheckpoint("best_lstm_model.keras", monitor="loss", save_best_only=True, mode="min")
early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min")

# Train model with an increased number of epochs
print("Starting model training...")
history = model.fit(data_generator, epochs=10, callbacks=[checkpoint, early_stopping])

# Plot training loss and accuracy to visualize training progress
print("Plotting training history...")
plt.plot(history.history['loss'], label='train_loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Define text generation function to demonstrate model's prediction capabilities
def generate_text(seed_text, next_words, model, max_sequence_len):
    print(f"Generating text with seed: '{seed_text}'")
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = tokenizer.index_word.get(np.argmax(predicted), "")
        seed_text += " " + output_word
    print(f"Generated text: '{seed_text}'")
    return seed_text

# Generate example text using a seed phrase
generate_text("To be or not", 5, model, max_sequence_len)

# Define function to compute cosine similarity between embeddings of word pairs
def get_cosine_similarity(word1, word2):
    vec1 = embeddings_index.get(word1)
    vec2 = embeddings_index.get(word2)
    if vec1 is not None and vec2 is not None:
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
    else:
        print(f"One of the words ('{word1}', '{word2}') does not have a pretrained embedding.")

# Calculate and display cosine similarity between sample word pairs
get_cosine_similarity("king", "queen")
get_cosine_similarity("man", "woman")

# Evaluate model performance on a sample batch for quick accuracy and loss check
print("Evaluating model on a sample batch...")
sample_X, sample_y = data_generator[0]
loss, accuracy = model.evaluate(sample_X, sample_y)
print(f"Sample Loss: {loss}, Sample Accuracy: {accuracy}")
