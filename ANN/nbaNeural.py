# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'all_seasons.csv'
players_df = pd.read_csv(data_path)

# Filter players within the 5-year window (e.g., 2015-2019)
selected_years = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
pool_df = players_df[players_df['season'].isin(selected_years)]

# Randomly select 100 players from the filtered pool
pool_df = pool_df.sample(n=100, random_state=42)

# Define a function to calculate scores for each role
def calculate_role_scores(df):
    # Calculate scores for each role based on relevant statistics
    df['scorer_score'] = df['pts'] * df['ts_pct']
    df['playmaker_score'] = df['ast'] * df['ast_pct']
    df['rebounder_score'] = df['reb'] * (df['oreb_pct'] + df['dreb_pct'])
    df['defender_score'] = df['net_rating'] + df['reb']
    df['utility_score'] = df['pts'] + df['reb'] + df['ast']
    return df

# Apply the scoring function
scored_df = calculate_role_scores(pool_df)

# Define real labels based on the highest score for each player
def assign_role(row):
    scores = {
        0: row['scorer_score'],    # Scorer
        1: row['playmaker_score'], # Playmaker
        2: row['rebounder_score'], # Rebounder
        3: row['defender_score'],  # Defender
        4: row['utility_score']    # Utility
    }
    return max(scores, key=scores.get)

# Assign labels based on the defined strategy
scored_df['role_label'] = scored_df.apply(assign_role, axis=1)
labels = scored_df['role_label'].values

# Extract features
features = scored_df[['scorer_score', 'playmaker_score', 'rebounder_score', 
                      'defender_score', 'utility_score']].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the neural network class
class OptimalTeamMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(OptimalTeamMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# Model parameters
input_size = X_train.shape[1]
hidden_sizes = [64, 32]
output_size = 5

# Instantiate the model
model = OptimalTeamMLP(input_size, hidden_sizes, output_size)

# Compute class weights to handle imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32)

# Define weighted loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=weights)  # Using weighted loss to address imbalance
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
loss_history = []
for epoch in range(epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss_history.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Plot loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = (predicted_labels == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, predicted_labels.numpy(), labels=np.arange(output_size))

# Display the confusion matrix manually
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

# Set tick positions and labels to match the expected classes
ax.set_xticks(np.arange(output_size))
ax.set_yticks(np.arange(output_size))
display_labels = ['Scorer', 'Playmaker', 'Rebounder', 'Defender', 'Utility']
ax.set_xticklabels(display_labels, rotation=45, ha="right")
ax.set_yticklabels(display_labels)

# Display counts on the confusion matrix cells
for i in range(output_size):
    for j in range(output_size):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.title('Confusion Matrix of Model Predictions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Selecting the optimal players based on model predictions
predicted_roles = model(torch.tensor(features_scaled, dtype=torch.float32))
_, role_predictions = torch.max(predicted_roles, 1)

# Create a DataFrame of selected players based on predictions
scored_df['predicted_role'] = role_predictions.numpy()

# Improved function to handle missing predictions for specific roles
def select_top_player(df, role, score_col):
    filtered_df = df[df['predicted_role'] == role].sort_values(by=score_col, ascending=False)
    if not filtered_df.empty:
        return filtered_df.iloc[0]
    else:
        print(f"No players predicted as role {role} ({score_col}).")
        return pd.Series({'player_name': 'No Player Selected', 'team_abbreviation': '-', 'scorer_score': 0, 
                          'playmaker_score': 0, 'rebounder_score': 0, 'defender_score': 0, 'utility_score': 0, 
                          'predicted_role': role})

# Select the top player for each predicted role with improved error handling
top_scorer = select_top_player(scored_df, 0, 'scorer_score')
top_playmaker = select_top_player(scored_df, 1, 'playmaker_score')
top_rebounder = select_top_player(scored_df, 2, 'rebounder_score')
top_defender = select_top_player(scored_df, 3, 'defender_score')
top_utility = select_top_player(scored_df, 4, 'utility_score')

# Create the final optimal team
optimal_team = pd.DataFrame([top_scorer, top_playmaker, top_rebounder, top_defender, top_utility])

# Display the optimal team
print("\nOptimal 5-Man Team Based on ANN Predictions:")
print(optimal_team[['player_name', 'team_abbreviation', 'scorer_score', 'playmaker_score', 
                    'rebounder_score', 'defender_score', 'utility_score', 'predicted_role']])