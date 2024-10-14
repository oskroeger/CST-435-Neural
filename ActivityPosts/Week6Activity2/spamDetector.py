import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
def load_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    return data['text'], data['label_num']

# Preprocess the email text
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Vectorize the preprocessed text using TF-IDF with bigrams and limited features
def vectorize_text(cleaned_emails):
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(cleaned_emails)
    return X, vectorizer

# Train a Naive Bayes or Logistic Regression classifier
def train_classifier(X_train, y_train, method="nb"):
    print(f"Training {method.upper()} classifier...")
    if method == "nb":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate the classifier
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    return y_pred

# Print two example emails and their predictions
def show_predictions(y_test, y_pred, emails, num_examples=2):
    print("\nSample Predictions:")
    for i in range(num_examples):
        print(f"\nEmail {i+1}:")
        print(f"Content: {emails.iloc[i]}")  # Access emails using iloc
        print(f"True Label: {'Spam' if y_test[i] == 1 else 'Ham'}")
        print(f"Predicted: {'Spam' if y_pred[i] == 1 else 'Ham'}")



# Main function
def main():
    # Load and preprocess data
    emails, labels = load_data('spam_ham_dataset.csv')
    
    print("Preprocessing emails...")
    cleaned_emails = [preprocess_text(email) for email in tqdm(emails)]
    
    # Vectorize the text
    X, vectorizer = vectorize_text(cleaned_emails)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Convert y_test to a NumPy array to prevent indexing issues
    y_test = y_test.to_numpy()

    # Train the classifier (switch between 'nb' for Naive Bayes and 'lr' for Logistic Regression)
    model = train_classifier(X_train, y_train, method="lr")
    
    # Evaluate the classifier
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Show sample predictions
    test_emails = emails.iloc[-len(y_test):].reset_index(drop=True)  # Reset index after slicing
    show_predictions(y_test, y_pred, test_emails)

if __name__ == "__main__":
    main()
