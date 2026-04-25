from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from model_utils import ModelManager

app = Flask(__name__, static_folder='.')
CORS(app)  # Allow frontend requests

# Global model manager
model_manager = None

def initialize_model():
    """Load or train the ML model using your CSV data"""
    global model_manager
    data_path = os.path.join(os.path.dirname(__file__), 'bucket_data', 'full_train.csv')
    if not os.path.exists(data_path):
        # fallback to combined data if full_train not found
        data_path = os.path.join(os.path.dirname(__file__), 'Bird_Migration_Data_with_Origin.csv')
    model_manager = ModelManager(data_path)
    model_manager.load_or_train()
    print("✅ Model ready")

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for migration prediction"""
    data = request.get_json()
    
    # Extract input features (same as frontend)
    temperature = float(data.get('temperature', 18))
    wind_speed = float(data.get('wind_speed', 22))
    precipitation = float(data.get('precipitation', 4.2))
    habitat = data.get('habitat', 'wetland')
    season = data.get('season', 'spring')
    food_abundance = float(data.get('food_abundance', 68))
    
    # Prepare feature vector
    features = {
        'temperature': temperature,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'habitat': habitat,
        'season': season,
        'food_abundance': food_abundance
    }
    
    # Get prediction from model
    result = model_manager.predict(features)
    
    return jsonify(result)

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000)