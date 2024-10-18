import pandas as pd
import spacy
import csv
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars

# Step 1: Load your full dataset CSV
csv_filename = 'covid19_tweets.csv'  # Replace with the path to your dataset CSV
df = pd.read_csv(csv_filename)

# Limit the dataset to the first 50,000 items
df = df.head(50000)

texts = df['text'].fillna('')  # Replace missing text values with an empty string

# Load spaCy's English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Tokenize and POS tag the 'text' column using spaCy, with progress bar
def tokenize_tweets(texts):
    processed_tweets = []
    for text in tqdm(texts, desc="Tokenizing and POS tagging"):
        doc = nlp(text)
        tokens = [(token.text, token.pos_) for token in doc]
        processed_tweets.append(tokens)
    return processed_tweets

# Tokenize only the first 50,000 items
tokenized_tweets = tokenize_tweets(texts)

# Step 2: Annotate and save the labeled data into a CSV file (Reduced to 10 sentences)
def manual_labeling_to_csv(tokenized_sentences, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "POS", "Label"])  # Header for the CSV file

        for sentence in tqdm(tokenized_sentences[:10], desc="Annotating Sentences"):  # Reduced to 10 sentences
            print("\nLabel the entities in the sentence:")
            for token, pos in sentence:
                print(f"\nToken: {token} (POS: {pos})")
                label = input("Enter label (O, B-MENTION, B-HASHTAG, B-URL, etc.): ").strip()
                writer.writerow([token, pos, label])
            writer.writerow([])  # Empty row between sentences

# Annotate and save the subset to CSV
csv_annotated_filename = 'annotated_subset.csv'
manual_labeling_to_csv(tokenized_tweets, csv_annotated_filename)

# Step 3: Load annotated data from CSV
def load_annotated_data_from_csv(csv_filename):
    sentences = []
    sentence = []

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if not row:  # Empty row means new sentence
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                token, pos, label = row
                sentence.append((token, pos, label))
    
    if sentence:
        sentences.append(sentence)

    return sentences

csv_annotated_filename = 'annotated_subset.csv'
annotated_data = load_annotated_data_from_csv(csv_annotated_filename)

# Step 4: Define the extract_features function (missing in previous code)
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

# Extract features and labels from the manually annotated data
def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def get_labels(sent):
    return [label for token, postag, label in sent]

# Extract features and labels from annotated data, using progress bar
X_train = [extract_features(s) for s in tqdm(annotated_data, desc="Extracting Features")]
y_train = [get_labels(s) for s in annotated_data]

# Train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=False
)

# Adding progress bar for training
print("Training the CRF model...")
crf.fit(X_train, y_train)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Evaluate the model
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
labels.remove('O')  # Remove 'O' from evaluation metrics

# Add `zero_division=0` to prevent warnings for undefined metrics
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, zero_division=0))