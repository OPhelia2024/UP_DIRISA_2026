from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

with open('UP_DIRISA_2026/src/model.pkl', 'rb') as f:
    model = joblib.load(f)


@app.route('/predict', methods=['POST'])
def predict_churn():
    """Endpoint for turnout prediction."""
    try:
        # Parse request data
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Get required feature columns
        feature_cols = ['Province', 
                        'Ward', 
                        'MunicipalityCode', 
                        'RegisteredVoters_prior',
                        'Turnout_prior',
                        'IsMetro',
                        'LogRegisteredVoters_prior',
                        'MunicipalityAvgTurnout_prior',
                        'SpoiltRatio_prior',
                        'SnapshotYear'
        ]
        X = df[feature_cols]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return jsonify({
            'predicted_turnout': prediction,
            'prediction_probability' : probability,
            'model_version': 'v1.3'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#Deploy using production WSGI servers like Gunicorn:
#gunicorn -w 4 -b 0.0.0.0:5000 src.model:app