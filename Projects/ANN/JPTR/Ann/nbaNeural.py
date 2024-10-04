# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'all_season.csv'
players_df = pd.read_csv(data_path)

# Filter players within the 5-year window (2015-2019)
selected_years = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
pool_df = players_df[players_df['season'].isin(selected_years)]

# Randomly select 100 players from the filtered pool
pool_df = pool_df.sample(n=100, random_state=42)

# Define a function to calculate scores for each role with simplified metrics
def calculate_role_scores(df):
    df['scorer_score'] = df['pts']
    df['playmaker_score'] = df['ast']
    df['rebounder_score'] = df['reb']
    df['defender_score'] = df['net_rating']
    df['utility_score'] = (df['pts'] + df['reb'] + df['ast']) / 3  # Average instead of sum
    df['impact_score'] = df['usg_pct'] * df['net_rating']
    df['size_factor'] = df['player_height'] * df['player_weight']
    return df

# Apply the simplified scoring function
scored_df = calculate_role_scores(pool_df)

# Extract balanced feature columns for each role model
features = scored_df[['scorer_score', 'playmaker_score', 'rebounder_score', 
                      'defender_score', 'utility_score', 'impact_score', 'size_factor']].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert to tensors for each role
X_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# Define the neural network class
class RoleSpecificMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RoleSpecificMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], 1)  # Single output for ranking
        )

    def forward(self, x):
        return self.layers(x)

# Model parameters
input_size = X_tensor.shape[1]
hidden_sizes = [128, 64]

# Function to train and rank players for a specific role and display a grid heatmap
def train_role_model(X_tensor, target_score, role_name):
    # Prepare target scores for the specific role
    y_tensor = torch.tensor(target_score, dtype=torch.float32).view(-1, 1)
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Instantiate the model
    model = RoleSpecificMLP(input_size, hidden_sizes)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Regression to rank players
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'{role_name} - Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Rank players by role score using trained model
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
        scored_df[f'{role_name}_predicted_score'] = predictions

    # Reshape predictions into a 10x10 grid for visualization
    grid_scores = np.array(predictions).reshape(10, 10)
    
    # Return the grid scores and candidate DataFrame for further processing
    return grid_scores, scored_df[[f'{role_name}_predicted_score', 'player_name', 'team_abbreviation', 'scorer_score', 
                                   'playmaker_score', 'rebounder_score', 'defender_score', 'utility_score']]

# Train separate models and collect all player predictions for each role
scorer_grid, scorer_candidates = train_role_model(X_tensor, scored_df['scorer_score'].values, 'Scorer')
playmaker_grid, playmaker_candidates = train_role_model(X_tensor, scored_df['playmaker_score'].values, 'Playmaker')
rebounder_grid, rebounder_candidates = train_role_model(X_tensor, scored_df['rebounder_score'].values, 'Rebounder')
defender_grid, defender_candidates = train_role_model(X_tensor, scored_df['defender_score'].values, 'Defender')
utility_grid, utility_candidates = train_role_model(X_tensor, scored_df['utility_score'].values, 'Utility')

# Combine all predictions into a single DataFrame
all_candidates = pd.concat([scorer_candidates, playmaker_candidates, rebounder_candidates, defender_candidates, utility_candidates])

# Ensure no duplicates and assign players based on best fit
assigned_players = set()
optimal_team = []

# Define roles and their candidates
role_order = [
    ('Scorer', scorer_candidates),
    ('Playmaker', playmaker_candidates),
    ('Rebounder', rebounder_candidates),
    ('Defender', defender_candidates),
    ('Utility', utility_candidates)
]

# Assign players based on highest available role score
selected_positions = {}
for role_name, candidates in role_order:
    candidates = candidates.sort_values(by=f'{role_name}_predicted_score', ascending=False)
    for _, player in candidates.iterrows():
        if player['player_name'] not in assigned_players:
            player['predicted_role'] = role_name
            optimal_team.append(player)
            assigned_players.add(player['player_name'])
            # Save the grid position of the selected player for annotation
            grid_index = candidates.index.get_loc(player.name)
            selected_positions[role_name] = (grid_index // 10, grid_index % 10, player['player_name'])
            break

# Create DataFrame of optimal team
optimal_team_df = pd.DataFrame(optimal_team)

# Display the optimal team
print("\nOptimal 5-Man Team Based on Separate Role-Specific Models (with Unique Selections):")
print(optimal_team_df[['player_name', 'team_abbreviation', 'scorer_score', 'playmaker_score', 
                       'rebounder_score', 'defender_score', 'utility_score', 'predicted_role']])

# Function to annotate and display heatmaps with the selected players
def plot_heatmap_with_selection(grid, role_name, candidates_df):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(label='Predicted Score')
    plt.title(f'{role_name} Heatmap of Predicted Scores')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.xticks(ticks=np.arange(10), labels=np.arange(1, 11))
    plt.yticks(ticks=np.arange(10), labels=np.arange(1, 11))
    
    # Annotate the heatmap with the selected player's name
    selected_player = optimal_team_df[optimal_team_df['predicted_role'] == role_name]
    for _, player in selected_player.iterrows():
        grid_index = candidates_df[candidates_df['player_name'] == player['player_name']].index[0]
        row, col = divmod(grid_index, 10)
        plt.text(col, row, player['player_name'], ha='center', va='center', 
                 color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.show()

# Plot each heatmap with the correct final selected player names
plot_heatmap_with_selection(scorer_grid, 'Scorer', scorer_candidates)
plot_heatmap_with_selection(playmaker_grid, 'Playmaker', playmaker_candidates)
plot_heatmap_with_selection(rebounder_grid, 'Rebounder', rebounder_candidates)
plot_heatmap_with_selection(defender_grid, 'Defender', defender_candidates)
plot_heatmap_with_selection(utility_grid, 'Utility', utility_candidates)