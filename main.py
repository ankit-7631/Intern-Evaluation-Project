import os
import warnings
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("=== Starting Fantasy Cricket Model Training ===")

# 1. Load Raw Datasets
df_matches = pd.read_csv('matches.csv')
df_deliveries = pd.read_csv('deliveries.csv')

# 2. Join Datasets
df_full = pd.merge(
    df_deliveries, 
    df_matches[['id', 'date', 'venue', 'toss_winner', 'toss_decision', 'winner']], 
    left_on='match_id', right_on='id', how='left'
)

# 3. Standardize Team Names
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiant': 'Rising Pune Supergiants'
}
df_full['batting_team'] = df_full['batting_team'].replace(team_mapping)
df_full['bowling_team'] = df_full['bowling_team'].replace(team_mapping)

# 4. Wickets & Extras Logic
df_full['is_legal_ball'] = df_full['extras_type'].apply(lambda x: 0 if x == 'wides' else 1)
bowler_wicket_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
df_full['is_bowler_wicket'] = df_full['dismissal_kind'].isin(bowler_wicket_types).astype(int)

# 5. Dream11 Fantasy Points Logic
df_full['batting_pts'] = df_full.apply(
    lambda r: r['batsman_runs'] * 1 + (1 if r['batsman_runs'] == 4 else 0) + (2 if r['batsman_runs'] == 6 else 0), axis=1
)
df_full['bowling_pts'] = df_full.apply(
    lambda r: (r['is_bowler_wicket'] * 25) + (8 if r['dismissal_kind'] in ['lbw', 'bowled'] else 0), axis=1
)

# 6. Aggregate Match Logs
batting_logs = df_full.groupby(['match_id', 'date', 'venue', 'batting_team', 'bowling_team', 'batter']).agg(
    bat_pts=('batting_pts', 'sum')
).reset_index().rename(columns={'batter': 'player_name', 'batting_team': 'team', 'bowling_team': 'opp_team'})

bowling_logs = df_full.groupby(['match_id', 'date', 'venue', 'bowling_team', 'batting_team', 'bowler']).agg(
    bowl_pts=('bowling_pts', 'sum')
).reset_index().rename(columns={'bowler': 'player_name', 'bowling_team': 'team', 'batting_team': 'opp_team'})

player_logs = pd.merge(
    batting_logs, bowling_logs, 
    on=['match_id', 'date', 'venue', 'player_name', 'team', 'opp_team'], 
    how='outer'
).fillna(0)

player_logs['actual_fantasy_pts'] = player_logs['bat_pts'] + player_logs['bowl_pts']
player_logs['date'] = pd.to_datetime(player_logs['date'])
player_logs = player_logs.sort_values(['player_name', 'date']).reset_index(drop=True)

# 7. High-Variance Feature Engineering
player_logs['career_avg_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(20.0)

player_logs['recent_peak_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).max()
).fillna(20.0)

player_logs['h2h_max_pts'] = player_logs.groupby(['player_name', 'opp_team'])['actual_fantasy_pts'].transform(
    lambda x: x.shift(1).expanding().max()
).fillna(player_logs['recent_peak_pts'])

player_logs['venue_max_pts'] = player_logs.groupby(['player_name', 'venue'])['actual_fantasy_pts'].transform(
    lambda x: x.shift(1).expanding().max()
).fillna(player_logs['recent_peak_pts'])

# 8. Train XGBoost Model
feature_cols = ['career_avg_pts', 'recent_peak_pts', 'h2h_max_pts', 'venue_max_pts']
X = player_logs[feature_cols]
y = player_logs['actual_fantasy_pts']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 9. Diagnostics Chart
plt.clf()
plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.set_theme(style="whitegrid")

y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.4, color='#0284c7', ax=axes[0])
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'Actual vs Predicted Points (MAE: {mae:.2f} | R²: {r2:.2f})')

residuals = y_test - y_pred_test
sns.histplot(residuals, kde=True, color='#0d9488', bins=30, ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Residual Error Distribution')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'model_accuracy_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# 10. Save Model and Processed Player Logs to PKL
artifacts = {
    'model': model,
    'player_logs': player_logs
}

joblib.dump(artifacts, 'model.pkl')
print(" SUCCESS: Model and player logs serialized to 'model.pkl'!")