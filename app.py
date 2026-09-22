import os
import re
import warnings
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load model artifacts
try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    player_logs = artifacts['player_logs']
except Exception as e:
    print(f"Error loading model.pkl: {e}")
    model, player_logs = None, None

def clean_venue_name(venue_str):
    if not venue_str:
        return ""
    # Remove text inside parentheses e.g. " (Chennai)"
    cleaned = re.sub(r'\s*\([^)]*\)', '', venue_str).strip()
    # Remove dots/periods e.g. "M.A." -> "MA"
    cleaned = cleaned.replace('.', '')
    return cleaned

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or player_logs is None:
            return jsonify({'error': 'Model artifacts failed to initialize on server.'}), 500

        data = request.get_json(force=True)
        player_name = data.get('player_name', '').strip()
        team_input = data.get('player_team', '').strip()
        opp_input = data.get('opp_team', '').strip()
        raw_venue = data.get('venue', '').strip()

        venue_input = clean_venue_name(raw_venue)

        # Look up player logs
        p_data = player_logs[player_logs['player_name'].str.contains(player_name, case=False, na=False, regex=False)]
        
        # GUARD 1: Missing Player
        if len(p_data) == 0:
            return jsonify({
                'player_name': player_name.upper() if player_name else "UNKNOWN",
                'team': team_input,
                'opp_team': opp_input,
                'venue': raw_venue,
                'projected_score': 0.0,
                'tier': "Invalid / Player Not Found",
                'stats': {'matches': 0, 'runs': 0, 'avg': 0.0, 'sr': 0.0, 'wickets': 0, 'economy': 0.0}
            })

        # GUARD 2: Player-Team Mismatch
        player_team_matches = p_data[p_data['team'].str.contains(team_input, case=False, na=False, regex=False)]
        if len(player_team_matches) == 0:
            return jsonify({
                'player_name': player_name.upper(),
                'team': team_input,
                'opp_team': opp_input,
                'venue': raw_venue,
                'career_pts': round(float(p_data['actual_fantasy_pts'].mean()), 1),
                'peak_pts': round(float(p_data['actual_fantasy_pts'].max()), 1),
                'projected_score': 0.0,
                'tier': f"0.0 Points ({player_name.title()} never played for {team_input})",
                'stats': {'matches': 0, 'runs': 0, 'avg': 0.0, 'sr': 0.0, 'wickets': 0, 'economy': 0.0}
            })

        # GUARD 3: Opposition Mismatch
        h2h_matches = p_data[p_data['opp_team'].str.contains(opp_input, case=False, na=False, regex=False)]
        if len(h2h_matches) == 0:
            return jsonify({
                'player_name': player_name.upper(),
                'team': team_input,
                'opp_team': opp_input,
                'venue': raw_venue,
                'career_pts': round(float(p_data['actual_fantasy_pts'].mean()), 1),
                'peak_pts': round(float(p_data['actual_fantasy_pts'].max()), 1),
                'projected_score': 0.0,
                'tier': f"0.0 Points (Never played against {opp_input})",
                'stats': {'matches': 0, 'runs': 0, 'avg': 0.0, 'sr': 0.0, 'wickets': 0, 'economy': 0.0}
            })

        # COMBINED MATCHUP QUERY: Opponent AND Cleaned Venue Name
        combo_matches = p_data[
            p_data['opp_team'].str.contains(opp_input, case=False, na=False, regex=False) &
            p_data['venue'].str.replace('.', '', regex=False).str.contains(venue_input, case=False, na=False, regex=False)
        ]

        combo_m_count = int(len(combo_matches))
        combo_runs = int(combo_matches['runs'].sum()) if 'runs' in combo_matches.columns else 0
        combo_balls_faced = int(combo_matches['balls_faced'].sum()) if 'balls_faced' in combo_matches.columns else 0
        combo_dismissals = int(combo_matches['dismissals'].sum()) if 'dismissals' in combo_matches.columns else 0

        combo_wkts = int(combo_matches['wickets'].sum()) if 'wickets' in combo_matches.columns else 0
        combo_balls_bowled = int(combo_matches['balls_bowled'].sum()) if 'balls_bowled' in combo_matches.columns else 0
        combo_runs_conceded = int(combo_matches['runs_conceded'].sum()) if 'runs_conceded' in combo_matches.columns else 0

        # Derived Metrics
        bat_avg = round(combo_runs / combo_dismissals, 1) if combo_dismissals > 0 else (float(combo_runs) if combo_runs > 0 else 0.0)
        bat_sr = round((combo_runs / combo_balls_faced) * 100, 1) if combo_balls_faced > 0 else 0.0
        bowling_econ = round((combo_runs_conceded / (combo_balls_bowled / 6.0)), 2) if combo_balls_bowled > 0 else 0.0

        # Features & Model Prediction
        career_pts = float(p_data['actual_fantasy_pts'].mean())
        peak_pts = float(p_data['actual_fantasy_pts'].max())
        h2h_pts = float(h2h_matches['actual_fantasy_pts'].max())
        
        venue_matches = p_data[p_data['venue'].str.replace('.', '', regex=False).str.contains(venue_input, case=False, na=False, regex=False)]
        venue_pts = float(venue_matches['actual_fantasy_pts'].max()) if len(venue_matches) > 0 else peak_pts

        user_features = pd.DataFrame([{
            'career_avg_pts': career_pts,
            'recent_peak_pts': peak_pts,
            'h2h_max_pts': h2h_pts,
            'venue_max_pts': venue_pts
        }])
        
        raw_pred = float(model.predict(user_features)[0])
        final_score = min(max(10.0, raw_pred), 150.0)

        if final_score >= 70:
            tier = "Excellent (Match Winner)"
        elif final_score >= 45:
            tier = "Good (Solid Pick)"
        elif final_score >= 25:
            tier = "Average (Filler)"
        else:
            tier = "Poor (Avoid)"

        return jsonify({
            'player_name': player_name.upper(),
            'team': team_input,
            'opp_team': opp_input,
            'venue': raw_venue,
            'career_pts': round(career_pts, 1),
            'peak_pts': round(peak_pts, 1),
            'h2h_pts': round(h2h_pts, 1),
            'venue_pts': round(venue_pts, 1),
            'projected_score': round(final_score, 1),
            'tier': tier,
            'stats': {
                'matches': combo_m_count,
                'runs': combo_runs,
                'avg': bat_avg,
                'sr': bat_sr,
                'wickets': combo_wkts,
                'economy': bowling_econ
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)