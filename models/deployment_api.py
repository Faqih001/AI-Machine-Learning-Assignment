#!/usr/bin/env python3
"""
Carbon Emission Prediction API
UN SDG 13 Climate Action Implementation
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model components at startup
with open('models/best_model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for carbon emission predictions"""
    try:
        # Get input data
        data = request.json

        # Convert to DataFrame and engineer features
        input_df = pd.DataFrame([data])

        # Feature engineering (same as training)
        input_df['GDP_per_energy'] = input_df['GDP_per_capita'] / input_df['Energy_consumption_per_capita']
        input_df['Population_density_proxy'] = input_df['Population'] * input_df['Urbanization_rate'] / 100
        input_df['Green_development_index'] = (
            input_df['Renewable_energy_pct'] * 0.4 +
            input_df['Forest_area_pct'] * 0.3 +
            input_df['Education_index'] * 100 * 0.3
        )
        input_df['Industrial_intensity'] = input_df['Industrial_production_index'] / input_df['GDP_per_capita']
        input_df['GDP_per_capita_log'] = np.log1p(input_df['GDP_per_capita'])
        input_df['Population_log'] = np.log1p(input_df['Population'])
        input_df['Energy_consumption_per_capita_log'] = np.log1p(input_df['Energy_consumption_per_capita'])

        # Select features and predict
        X = input_df[feature_names]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        return jsonify({
            'predicted_co2_emissions': float(prediction),
            'status': 'success',
            'sdg_target': 'SDG 13: Climate Action'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'carbon_emission_predictor'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
