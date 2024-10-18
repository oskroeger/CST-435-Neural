# Required imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import random

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Download NLTK resources if not already present
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load the IMDB dataset and store it in a DataFrame
# We choose the IMDB dataset as it is suitable for binary sentiment analysis (positive/negative)
print("Loading the IMDB dataset...")
dataset = load_dataset('imdb')  # Loading the IMDB dataset using the Huggingface datasets library
df = pd.DataFrame(dataset['train'])  # Store the dataset in a DataFrame

# Step 2: Handle missing values by dropping any rows with NaN values
df.dropna(inplace=True)

# Step 3: Perform a descriptive statistical analysis by counting positive and negative reviews
# This step helps us understand the class distribution
sns.countplot(x='label', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.show()

# Step 4: Define the text preprocessing function
# This function removes punctuation, digits, converts to lowercase, removes stopwords, and lemmatizes words.
# These are important steps in preparing the text data for machine learning models.
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords and lemmatize the remaining words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    return text

# Step 5: Preprocess the text data
# We apply the preprocessing function to the entire dataset using a progress bar
print("Preprocessing text data...")
df['processed_text'] = [preprocess_text(text) for text in tqdm(df['text'])]

# Step 6: Convert the text data into numerical features using TF-IDF Vectorizer
# TF-IDF assigns a score to each word based on how often it appears in a document, helping us build a feature matrix
tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1, 3))  # Unigrams, Bigrams, and Trigrams
X = tfidf.fit_transform(df['processed_text']).toarray()  # Convert processed text into TF-IDF feature vectors
y = df['label']  # Labels (0 = Negative, 1 = Positive)

# Step 7: Split the data into training and testing sets (80:20 split)
# This is a standard practice to evaluate model performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build and train a Logistic Regression model
# Logistic Regression is a binary classification algorithm, commonly used in sentiment analysis
# The C parameter is adjusted to control regularization (a form of model tuning)
model = LogisticRegression(max_iter=2000, C=0.7)

print("Training the Logistic Regression model...")
model.fit(X_train, y_train)  # Fit the model on the training data

# Step 9: Evaluate the model by calculating the accuracy on the test set
# We predict sentiment labels for the test set and compute the accuracy score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy:.4f}")

# Step 10: Display real reviews from the dataset along with their predicted sentiment
# This step simulates making predictions on real-world inputs
def display_balanced_reviews_with_predictions(df, model, tfidf, num_samples=6):
    # Separate the reviews into positive and negative groups
    positive_reviews = df[df['label'] == 1]
    negative_reviews = df[df['label'] == 0]
    
    # Randomly sample half positive and half negative reviews
    pos_sample = positive_reviews.sample(num_samples // 2, random_state=42)
    neg_sample = negative_reviews.sample(num_samples // 2, random_state=42)
    
    # Combine the samples into a single DataFrame
    sampled_reviews = pd.concat([pos_sample, neg_sample])
    
    # Preprocess the sampled reviews
    processed_reviews = [preprocess_text(text) for text in sampled_reviews['text']]
    review_features = tfidf.transform(processed_reviews).toarray()
    
    # Predict sentiment for the sampled reviews
    predictions = model.predict(review_features)
    
    # Print each review and its predicted sentiment
    for idx, (text, pred) in enumerate(zip(sampled_reviews['text'], predictions)):
        sentiment_label = 'Positive' if pred == 1 else 'Negative'
        print(f"Review {idx + 1}:\n{text}\nPredicted Sentiment: {sentiment_label}\n")

# Display 6 real reviews (3 positive, 3 negative) along with their predicted sentiment
display_balanced_reviews_with_predictions(df, model, tfidf, num_samples=6)

# Step 11: Evaluate the model performance using a confusion matrix
# The confusion matrix shows how well the model distinguishes between the two classes (positive and negative)
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 12: Print the classification report for additional performance metrics
# This report includes precision, recall, and F1-score, which are important for understanding model performance
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print final accuracy of the model
print(f"Accuracy: {accuracy:.4f}")
