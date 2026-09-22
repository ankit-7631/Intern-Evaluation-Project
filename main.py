import os
import warnings
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

print("=== Starting Fantasy Cricket Model Training ===")

# 1. Load Raw Datasets
df_matches = pd.read_csv('matches.csv')
df_deliveries = pd.read_csv('deliveries.csv')

# Merge match context (date, venue) into delivery logs
df_full = pd.merge(
    df_deliveries, 
    df_matches[['id', 'date', 'venue']], 
    left_on='match_id', right_on='id', how='left'
)

# Standardize Team Names
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Rising Pune Supergiant': 'Rising Pune Supergiants'
}
df_full['batting_team'] = df_full['batting_team'].replace(team_mapping)
df_full['bowling_team'] = df_full['bowling_team'].replace(team_mapping)

# 2. Extract Dismissals, Wickets, and Ball Counts
bowler_wicket_types = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
df_full['is_bowler_wicket'] = df_full['dismissal_kind'].isin(bowler_wicket_types).astype(int)
df_full['is_dismissal'] = df_full['dismissal_kind'].notna().astype(int)

df_full['is_legal_ball'] = (~df_full['extras_type'].isin(['wides', 'noballs'])).astype(int)
df_full['is_batter_ball'] = (df_full['extras_type'] != 'wides').astype(int)

# 3. Aggregate Batting Match Logs
batting_logs = df_full.groupby(['match_id', 'date', 'venue', 'batting_team', 'bowling_team', 'batter']).agg(
    runs=('batsman_runs', 'sum'),
    balls_faced=('is_batter_ball', 'sum'),
    dismissals=('is_dismissal', 'sum')
).reset_index().rename(columns={'batter': 'player_name', 'batting_team': 'team', 'bowling_team': 'opp_team'})

# 4. Aggregate Bowling Match Logs
bowling_logs = df_full.groupby(['match_id', 'date', 'venue', 'bowling_team', 'batting_team', 'bowler']).agg(
    wickets=('is_bowler_wicket', 'sum'),
    balls_bowled=('is_legal_ball', 'sum'),
    runs_conceded=('total_runs', 'sum')
).reset_index().rename(columns={'bowler': 'player_name', 'bowling_team': 'team', 'batting_team': 'opp_team'})

# 5. Merge Batting and Bowling Logs
player_logs = pd.merge(
    batting_logs, bowling_logs, 
    on=['match_id', 'date', 'venue', 'player_name', 'team', 'opp_team'], 
    how='outer'
).fillna(0)

# Calculate Match Fantasy Points
player_logs['actual_fantasy_pts'] = (player_logs['runs'] * 1.0) + (player_logs['wickets'] * 25.0)
player_logs['date'] = pd.to_datetime(player_logs['date'])
player_logs = player_logs.sort_values(['player_name', 'date']).reset_index(drop=True)

# 6. Feature Aggregations
player_logs['career_avg_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform('mean')
player_logs['recent_peak_pts'] = player_logs.groupby('player_name')['actual_fantasy_pts'].transform('max')

h2h_max = player_logs.groupby(['player_name', 'opp_team'])['actual_fantasy_pts'].max().reset_index().rename(columns={'actual_fantasy_pts': 'h2h_max_pts'})
player_logs = player_logs.merge(h2h_max, on=['player_name', 'opp_team'], how='left')

venue_max = player_logs.groupby(['player_name', 'venue'])['actual_fantasy_pts'].max().reset_index().rename(columns={'actual_fantasy_pts': 'venue_max_pts'})
player_logs = player_logs.merge(venue_max, on=['player_name', 'venue'], how='left')

player_logs.fillna(0, inplace=True)

# 7. Train Lightweight Scikit-Learn Model
feature_cols = ['career_avg_pts', 'recent_peak_pts', 'h2h_max_pts', 'venue_max_pts']
X = player_logs[feature_cols]
y = player_logs['actual_fantasy_pts']

model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=6, random_state=42)
model.fit(X, y)

# 8. Save Artifacts
artifacts = {
    'model': model,
    'player_logs': player_logs
}

joblib.dump(artifacts, 'model.pkl')
print("SUCCESS: Model and player logs serialized to 'model.pkl'!")