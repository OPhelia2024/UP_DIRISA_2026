from flask import Flask, request, jsonify
import pickle
import pandas as pd
from src.features import ChurnFeatureEngineer

app = Flask(__name__)

with open('UP_DIRISA_2026/src/model.pkl', 'rb') as f:
    model = pickle.load(f)

feature_engineer = ChurnFeatureEngineer()    

@app.route('/predict', methods=['POST'])
def predict_churn():
    """Endpoint for churn prediction."""
    try:
        # Parse request data
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Engineer features
        df_features = feature_engineer.engineer_features(df)
        
        # Get required feature columns
        feature_cols = ['tenure_months', 'avg_monthly_usage', 
                       'days_since_last_activity', 'total_spend']
        X = df_features[feature_cols]
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return jsonify({
            'customer_id': data['customer_id'],
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability),
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