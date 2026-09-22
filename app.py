import warnings
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')

app = Flask(__name__)

# 1. Load Pre-trained Artifacts from PKL
print("Loading PKL artifacts...")
try:
    artifacts = joblib.load('model.pkl')
    model = artifacts['model']
    player_logs = artifacts['player_logs']
    print("PKL Model loaded successfully!")
except Exception as e:
    print(f"Error loading model.pkl: {e}")

# 2. Root Route (Serves templates/index.html)
@app.route('/')
def home():
    return render_template('index.html')

# 3. Prediction API Route with Validation Guards
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        player_name = data.get('player_name', '').strip()
        team_input = data.get('player_team', '').strip()
        opp_input = data.get('opp_team', '').strip()
        venue_input = data.get('venue', '').strip()

        # Look up player in historical database
        p_data = player_logs[player_logs['player_name'].str.contains(player_name, case=False, na=False, regex=False)]
        
        # GUARD 1: Player not found in database at all
        if len(p_data) == 0:
            return jsonify({
                'player_name': player_name.upper() if player_name else "UNKNOWN",
                'team': team_input,
                'opp_team': opp_input,
                'venue': venue_input,
                'career_pts': 0.0,
                'peak_pts': 0.0,
                'h2h_pts': 0.0,
                'venue_pts': 0.0,
                'projected_score': 0.0,
                'tier': "Invalid / Player Not Found"
            })

        # GUARD 2: Check if player ever played for the selected team
        player_team_matches = p_data[p_data['team'].str.contains(team_input, case=False, na=False, regex=False)]
        if len(player_team_matches) == 0:
            career_pts = float(p_data['actual_fantasy_pts'].mean())
            peak_pts = float(p_data['actual_fantasy_pts'].max())
            
            return jsonify({
                'player_name': player_name.upper(),
                'team': team_input,
                'opp_team': opp_input,
                'venue': venue_input,
                'career_pts': round(career_pts, 1),
                'peak_pts': round(peak_pts, 1),
                'h2h_pts': 0.0,
                'venue_pts': 0.0,
                'projected_score': 0.0,
                'tier': f"0.0 Points ({player_name.title()} never played for {team_input})"
            })

        # GUARD 3: Check if player ever played against the selected opposition
        h2h_matches = p_data[p_data['opp_team'].str.contains(opp_input, case=False, na=False, regex=False)]
        if len(h2h_matches) == 0:
            career_pts = float(p_data['actual_fantasy_pts'].mean())
            peak_pts = float(p_data['actual_fantasy_pts'].max())
            
            return jsonify({
                'player_name': player_name.upper(),
                'team': team_input,
                'opp_team': opp_input,
                'venue': venue_input,
                'career_pts': round(career_pts, 1),
                'peak_pts': round(peak_pts, 1),
                'h2h_pts': 0.0,
                'venue_pts': 0.0,
                'projected_score': 0.0,
                'tier': f"0.0 Points (Never played against {opp_input})"
            })

        # VALID MATCHUP PROCESSING
        career_pts = float(p_data['actual_fantasy_pts'].mean())
        peak_pts = float(p_data['actual_fantasy_pts'].max())
        h2h_pts = float(h2h_matches['actual_fantasy_pts'].max())
            
        venue_matches = p_data[p_data['venue'].str.contains(venue_input, case=False, na=False, regex=False)]
        venue_pts = float(venue_matches['actual_fantasy_pts'].max()) if len(venue_matches) > 0 else peak_pts

        user_features = pd.DataFrame([{
            'career_avg_pts': career_pts,
            'recent_peak_pts': peak_pts,
            'h2h_max_pts': h2h_pts,
            'venue_max_pts': venue_pts
        }])
        
        raw_pred = float(model.predict(user_features)[0])
        
        if peak_pts > 50:
            scaled_pred = raw_pred * (1.0 + ((peak_pts - 50) / 100))
        else:
            scaled_pred = raw_pred
            
        final_score = max(10.0, scaled_pred)

        if final_score >= 70:
            tier = "Excellent (Match Winner / Captain Choice)"
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
            'venue': venue_input,
            'career_pts': round(career_pts, 1),
            'peak_pts': round(peak_pts, 1),
            'h2h_pts': round(h2h_pts, 1),
            'venue_pts': round(venue_pts, 1),
            'projected_score': round(final_score, 1),
            'tier': tier
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Keep app.run inside the __main__ check so Vercel can handle WSGI serving
if __name__ == '__main__':
    app.run(debug=True, port=5000)