import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set non-interactive Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 1. Load raw datasets
df_matches = pd.read_csv('matches.csv')
df_deliveries = pd.read_csv('deliveries.csv')

# 2. Join datasets on match ID
df_full = pd.merge(
    df_deliveries, 
    df_matches[['id', 'date', 'venue', 'toss_winner', 'toss_decision', 'winner']], 
    left_on='match_id', 
    right_on='id', 
    how='left'
)

# 3. Standardize Franchise Team Names
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiant': 'Rising Pune Supergiants'
}

df_full['batting_team'] = df_full['batting_team'].replace(team_mapping)
df_full['bowling_team'] = df_full['bowling_team'].replace(team_mapping)

# 4. Ball-by-Ball Wicket and Legal Delivery Logic
df_full['is_legal_ball'] = df_full['extras_type'].apply(lambda x: 0 if x == 'wides' else 1)
bowler_wicket_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
df_full['is_bowler_wicket'] = df_full['dismissal_kind'].isin(bowler_wicket_types).astype(int)

# 5. Dream11 Points Logic
def calc_batting_pts(row):
    runs = row['batsman_runs']
    fours = 1 if (runs == 4) else 0
    sixes = 1 if (runs == 6) else 0
    return runs * 1 + fours * 1 + sixes * 2

def calc_bowling_pts(row):
    wickets = row['is_bowler_wicket']
    lbw_bowled = 1 if row['dismissal_kind'] in ['lbw', 'bowled'] else 0
    return wickets * 25 + lbw_bowled * 8

df_full['batting_pts'] = df_full.apply(calc_batting_pts, axis=1)
df_full['bowling_pts'] = df_full.apply(calc_bowling_pts, axis=1)

# 6. Aggregate Match Logs
player_logs = df_full.groupby(['match_id', 'date', 'venue', 'batting_team', 'bowling_team', 'batter']).agg(
    actual_fantasy_pts=('batting_pts', 'sum'),
    runs=('batsman_runs', 'sum'),
    balls=('is_legal_ball', 'sum'),
    fours=('batsman_runs', lambda x: (x == 4).sum()),
    sixes=('batsman_runs', lambda x: (x == 6).sum())
).reset_index().rename(columns={'batter': 'player_name', 'batting_team': 'team', 'bowling_team': 'opp_team'})

player_logs['date'] = pd.to_datetime(player_logs['date'])
player_logs = player_logs.sort_values(['player_name', 'date']).reset_index(drop=True)

# 7. High-Variance Feature Engineering
player_logs['career_avg_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform(lambda x: x.shift(1).expanding().mean()).fillna(20.0)
player_logs['recent_peak_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).fillna(20.0)
player_logs['h2h_max_pts'] = player_logs.groupby(['player_name', 'opp_team'])['actual_fantasy_pts'].transform(lambda x: x.shift(1).expanding().max()).fillna(player_logs['recent_peak_pts'])
player_logs['venue_max_pts'] = player_logs.groupby(['player_name', 'venue'])['actual_fantasy_pts'].transform(lambda x: x.shift(1).expanding().max()).fillna(player_logs['recent_peak_pts'])

# 8. Uncapped Features (X) & Target (y)
feature_cols = ['career_avg_pts', 'recent_peak_pts', 'h2h_max_pts', 'venue_max_pts']
X = player_logs[feature_cols]
y = player_logs['actual_fantasy_pts']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Scaled Predictor Function to stretch upper variance
def predict_uncapped_score(input_df):
    raw_pred = model.predict(input_df)[0]
    peak_val = input_df['recent_peak_pts'].iloc[0]
    
    if peak_val > 50:
        multiplier = 1.0 + ((peak_val - 50) / 100)
        scaled_pred = raw_pred * multiplier
    else:
        scaled_pred = raw_pred
        
    return max(10.0, scaled_pred)

# 9. Model Evaluation & Explicit Graph Generation
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 50)
print("       MODEL ACCURACY EVALUATION METRICS")
print("=" * 50)
print(f" • Mean Absolute Error (MAE): {mae:.2f} points")
print(f" • Root Mean Squared Error (RMSE): {rmse:.2f} points")
print(f" • R-squared (R²) Score: {r2:.2f}")
print("=" * 50 + "\n")

plt.clf()
plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.set_theme(style="whitegrid")

# Panel 1: Actual vs. Predicted Points
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.4, color='#0284c7', ax=axes[0])
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Ideal 1:1 Line')
axes[0].set_title(f'Actual vs Predicted Points\n(MAE: {mae:.2f} | R²: {r2:.2f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Fantasy Points Scored', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Model Predicted Points', fontsize=10, fontweight='bold')
axes[0].legend(loc='upper left')

# Panel 2: Residual Error Distribution
residuals = y_test - y_pred_test
sns.histplot(residuals, kde=True, color='#0d9488', bins=30, ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
axes[1].set_title(f'Prediction Residuals Distribution\n(RMSE: {rmse:.2f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residual Error (Actual - Predicted)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=10, fontweight='bold')
axes[1].legend(loc='upper right')

plt.tight_layout()

save_path = os.path.join(os.getcwd(), 'model_accuracy_diagnostics.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"IMAGE SAVED AT: {save_path}\n")

# 10. Interactive Input Loop
while True:
    print("=" * 55)
    print("   ADVANCED DREAM11 PLAYER PERFORMANCE PREDICTOR")
    print("=" * 55)
    
    player_input = input("Enter Player Name (e.g. V Kohli) [type 'exit' to quit]: ").strip()
    if player_input.lower() == 'exit':
        break
        
    team_input = input("Enter Player's Team (e.g. Royal Challengers Bengaluru): ").strip()
    opp_input = input("Enter Opposition Team (e.g. Punjab Kings): ").strip()
    venue_input = input("Enter Match Venue (e.g. M Chinnaswamy Stadium): ").strip()
    
    p_data = player_logs[player_logs['player_name'].str.contains(player_input, case=False, na=False, regex=False)]
    
    if len(p_data) == 0:
        print(f"\nWarning: '{player_input}' not found in database. Using defaults.\n")
        career_pts, peak_pts, h2h_pts, venue_pts = 20.0, 20.0, 20.0, 20.0
    else:
        career_pts = p_data['actual_fantasy_pts'].mean()
        peak_pts = p_data['actual_fantasy_pts'].max()
        
        h2h_matches = p_data[p_data['opp_team'].str.contains(opp_input, case=False, na=False, regex=False)]
        h2h_pts = h2h_matches['actual_fantasy_pts'].max() if len(h2h_matches) > 0 else peak_pts
            
        venue_matches = p_data[p_data['venue'].str.contains(venue_input, case=False, na=False, regex=False)]
        venue_pts = venue_matches['actual_fantasy_pts'].max() if len(venue_matches) > 0 else peak_pts

    # Aligned feature column names matching model training input exactly
    user_features = pd.DataFrame([{
        'career_avg_pts': career_pts,
        'recent_peak_pts': peak_pts,
        'h2h_max_pts': h2h_pts,
        'venue_max_pts': venue_pts
    }])
    
    final_projection = predict_uncapped_score(user_features)

    if final_projection >= 70:
        tier = "Excellent (Match Winner / Captain Choice)"
    elif final_projection >= 45:
        tier = "Good (Solid Pick)"
    elif final_projection >= 25:
        tier = "Average (Filler)"
    else:
        tier = "Poor (Avoid)"

    print("\n" + "-" * 50)
    print(f" PREDICTION RESULT FOR: {player_input.upper()}")
    print(f" Match: {team_input} vs {opp_input}")
    print(f" Venue: {venue_input}")
    print(f" • Career Baseline Points : {career_pts:.1f}")
    print(f" • Peak Form Potential   : {peak_pts:.1f}")
    print(f" • Head-to-Head Peak     : {h2h_pts:.1f}")
    print(f" • Venue Peak            : {venue_pts:.1f}")
    print(f" • Projected Score       : {final_projection:.1f} Points")
    print(f" • Performance Tier      : {tier}")
    print("-" * 50 + "\n")