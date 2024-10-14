import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Variables to set the number of epochs and samples
num_epochs = 3  # Reduce epochs for faster testing
num_samples = 1000  # Limit to 1000 samples for faster processing

# Step 1: Load dataset and model tokenizer
print("Loading the IMDB dataset...")
dataset = load_dataset('imdb')

print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Data Exploration
train_df = pd.DataFrame(dataset["train"])
sns.countplot(x='label', data=train_df)
plt.title('Class distribution')
plt.show()

# Step 2: Add sentiment labels on a scale from -3 to 3 using VADER
print("Initializing VADER sentiment analyzer...")
analyzer = SentimentIntensityAnalyzer()

def score_to_label(score):
    if score >= 0.6:
        return 6  # Shift from 3 to 6
    elif 0.2 <= score < 0.6:
        return 5  # Shift from 2 to 5
    elif 0 <= score < 0.2:
        return 4  # Shift from 1 to 4
    elif -0.2 <= score < 0:
        return 3  # Shift from 0 to 3
    elif -0.6 <= score < -0.2:
        return 2  # Shift from -1 to 2
    elif -1 <= score < -0.6:
        return 1  # Shift from -2 to 1
    else:
        return 0  # Shift from -3 to 0

def add_sentiment_labels(dataset):
    dataset_with_sentiments = []
    print_interval = 100  # Print progress every 100 reviews
    for i, review in enumerate(dataset['text']):
        score = analyzer.polarity_scores(review)['compound']
        label = score_to_label(score)
        dataset_with_sentiments.append({'text': review, 'label': label})
        if i % print_interval == 0:
            print(f"Processed {i} reviews")
    return dataset_with_sentiments

print("Labeling training data (limited to 1000 samples for faster processing)...")
train_data_with_sentiments = add_sentiment_labels(dataset['train'].select(range(num_samples)))  # Limit to 1000 samples
test_data_with_sentiments = add_sentiment_labels(dataset['test'].select(range(num_samples)))    # Limit to 1000 samples

# Convert into pandas DataFrame for easier processing
train_df = pd.DataFrame(train_data_with_sentiments)
test_df = pd.DataFrame(test_data_with_sentiments)

# Step 3: Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenizing training data...")
tokenized_train = tokenizer(list(train_df['text']), padding=True, truncation=True, max_length=128)  # Reduced max_length

print("Tokenizing test data...")
tokenized_test = tokenizer(list(test_df['text']), padding=True, truncation=True, max_length=128)  # Reduced max_length

# Custom Dataset class for PyTorch
class SentimentDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long).clone().detach(),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long).clone().detach(),
            'labels': self.labels[idx].clone().detach()  # Avoid copying warnings
        }

# Prepare datasets for PyTorch
train_dataset = SentimentDataset(tokenized_train, train_df['label'].values)
test_dataset = SentimentDataset(tokenized_test, test_df['label'].values)

# Step 4: Load pre-trained DistilBERT model with 7 output labels (for -3 to 3 scale)
print("Loading the pre-trained DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)

# Step 5: Define training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    "test_trainer",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    per_device_train_batch_size=32,  # Increase batch size to use more memory
    per_device_eval_batch_size=32,
    num_train_epochs=num_epochs,
    weight_decay=0,  # Disable weight decay for speed
    logging_dir='./logs',
    logging_steps=500,  # Log less frequently
    fp16=False,  # Disable mixed precision since we're on CPU
    gradient_checkpointing=False  # Disable gradient checkpointing for speed
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Step 6: Train the model
trainer.train()
print("Training complete.")

# Step 7: Evaluation
print("Evaluating the model...")
predictions = trainer.predict(test_dataset)

# Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(test_df['label'], predictions.predictions.argmax(-1))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve (for demonstration purposes)
fpr, tpr, _ = roc_curve(test_df['label'], predictions.predictions[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Ensure the model is moved to CPU
model.to(torch.device('cpu'))

# Step 8: Inference on multiple samples
sample_texts = [
    "This is a fantastic movie. I really enjoyed it.",
    "The plot was boring and the acting was terrible.",
    "It was okay, not the best but not the worst.",
    "Amazing cinematography, but the story lacked depth.",
    "I would not recommend this movie to anyone."
]

print(f"Making predictions on the following samples:")
for text in sample_texts:
    print(f"- {text}")

# Tokenize the batch of input samples
sample_inputs = tokenizer(sample_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Move inputs to CPU
sample_inputs = {k: v.to(torch.device('cpu')) for k, v in sample_inputs.items()}

# Make predictions
with torch.no_grad():  # Disable gradient calculations for inference
    predictions = model(**sample_inputs)

# Get logits (raw model outputs before softmax)
logits = predictions.logits

# Calculate confidence for each sample using softmax
probabilities = F.softmax(logits, dim=-1)

# Adjust predicted sentiment scores from probabilities
def logits_to_sentiment_scale(logits):
    pos_confidence = logits[1]  # Confidence for positive sentiment
    neg_confidence = logits[0]  # Confidence for negative sentiment
    
    # If positive sentiment is more confident, map to positive scale (1 to 3)
    if pos_confidence > neg_confidence:
        confidence_score = pos_confidence - neg_confidence
        if confidence_score >= 0.6:
            return 3  # Very confident positive
        elif confidence_score >= 0.3:
            return 2  # Moderately confident positive
        else:
            return 1  # Slightly positive
        
    # If negative sentiment is more confident, map to negative scale (-1 to -3)
    else:
        confidence_score = neg_confidence - pos_confidence
        if confidence_score >= 0.6:
            return -3  # Very confident negative
        elif confidence_score >= 0.3:
            return -2  # Moderately confident negative
        else:
            return -1  # Slightly negative

# Convert logits to sentiment scores on a -3 to 3 scale
predicted_sentiments = [logits_to_sentiment_scale(logit) for logit in probabilities]

# Output sentiment predictions for each sample
for text, sentiment in zip(sample_texts, predicted_sentiments):
    print(f"Text: '{text}'\nPredicted sentiment score: {sentiment}\n")
