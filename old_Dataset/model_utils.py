import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class ModelManager:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.habitat_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.species_mapping = {}   # for guild / species name mapping
        self.model_path = 'migration_model.pkl'
        self.scaler_path = 'scaler.pkl'
        self.encoders_path = 'encoders.pkl'
    
    def load_or_train(self):
        """Load pre-trained model if exists, otherwise train from CSV"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            print("📦 Loading pre-trained model...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.encoders_path, 'rb') as f:
                encoders = pickle.load(f)
                self.habitat_encoder = encoders['habitat']
                self.season_encoder = encoders['season']
                self.species_mapping = encoders.get('species_mapping', {})
            return
        
        print("🚀 Training new model from CSV data...")
        self._train_from_csv()
        # Save model and encoders
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.encoders_path, 'wb') as f:
            pickle.dump({
                'habitat': self.habitat_encoder,
                'season': self.season_encoder,
                'species_mapping': self.species_mapping
            }, f)
        print("✅ Model saved for future use.")
    
    def _train_from_csv(self):
        """Load CSV, engineer features, train Random Forest"""
        df = pd.read_csv(self.data_path)
        
        # --- Feature engineering (adjust column names to match your actual CSV) ---
        # Typical columns you might have: temperature, wind_speed, precipitation,
        # habitat_type, season, food_index, migration_probability, species, guild
        # If column names differ, adapt here.
        
        # Map habitat and season to numeric
        if 'habitat' in df.columns:
            self.habitat_encoder.fit(df['habitat'].astype(str))
            df['habitat_encoded'] = self.habitat_encoder.transform(df['habitat'].astype(str))
        else:
            # fallback: create dummy column
            df['habitat_encoded'] = 0
            self.habitat_encoder.fit(['wetland', 'forest', 'grassland', 'urban', 'coastal'])
        
        if 'season' in df.columns:
            self.season_encoder.fit(df['season'].astype(str))
            df['season_encoded'] = self.season_encoder.transform(df['season'].astype(str))
        else:
            df['season_encoded'] = 0
            self.season_encoder.fit(['spring', 'summer', 'autumn', 'winter'])
        
        # Define feature columns (adjust to actual column names in your CSV)
        feature_cols = []
        if 'temperature' in df.columns:
            feature_cols.append('temperature')
        else:
            df['temperature'] = 18  # placeholder
            feature_cols.append('temperature')
        
        if 'wind_speed' in df.columns:
            feature_cols.append('wind_speed')
        else:
            df['wind_speed'] = 22
            feature_cols.append('wind_speed')
        
        if 'precipitation' in df.columns:
            feature_cols.append('precipitation')
        else:
            df['precipitation'] = 4
            feature_cols.append('precipitation')
        
        if 'food_abundance' in df.columns:
            feature_cols.append('food_abundance')
        else:
            df['food_abundance'] = 68
            feature_cols.append('food_abundance')
        
        feature_cols.extend(['habitat_encoded', 'season_encoded'])
        
        # Target variable: we need migration probability (0-1 or 0-100)
        # If your CSV has 'migration_probability', use it.
        # Otherwise, create synthetic target based on other columns for demo.
        if 'migration_probability' in df.columns:
            y = df['migration_probability'].values
        else:
            # Heuristic: migration probability = f(temp, wind, food) – for demonstration only
            # In real case, you should have a real target.
            np.random.seed(42)
            y = (df['temperature'] - 10) / 30 + df['wind_speed'] / 100 + df['food_abundance'] / 200
            y = np.clip(y, 0, 1)
        
        X = df[feature_cols].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train regression model for probability
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Also store species/guild mapping for nice output (if available)
        if 'species' in df.columns:
            species_list = df['species'].dropna().unique()
            self.species_mapping = {i: sp for i, sp in enumerate(species_list)}
        else:
            self.species_mapping = {
                0: "Tree Swallow",
                1: "Barn Swallow",
                2: "Canada Goose",
                3: "Ruby-throated Hummingbird"
            }
    
    def predict(self, input_features):
        """Make prediction for a single set of inputs"""
        # Encode categorical inputs
        habitat_enc = self.habitat_encoder.transform([input_features['habitat']])[0]
        season_enc = self.season_encoder.transform([input_features['season']])[0]
        
        # Build feature vector (must match training order)
        features = np.array([[
            input_features['temperature'],
            input_features['wind_speed'],
            input_features['precipitation'],
            input_features['food_abundance'],
            habitat_enc,
            season_enc
        ]])
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict migration probability
        prob = self.model.predict(features_scaled)[0]
        prob_percent = round(prob * 100, 1)
        
        # Determine species / guild based on probability and habitat (rule-based + model)
        # You can enhance this using a multi-output classifier if your data supports.
        if prob > 0.75:
            risk = "🔥 High migration activity — excellent conditions, peak passage expected"
            cons = "⚠️ CRITICAL: Reduce light pollution & human disturbance"
        elif prob > 0.55:
            risk = "🌤️ Moderate movement, favorable conditions for migration"
            cons = "🌿 Priority window: ensure habitat connectivity"
        elif prob > 0.35:
            risk = "🌿 Low migration intensity, birds likely resting or foraging"
            cons = "📊 Good time for citizen science monitoring"
        else:
            risk = "❄️ Minimal migration expected, conditions not favorable"
            cons = "🍃 Low activity: ideal for habitat restoration"
        
        # Select species based on habitat and season (simple mapping)
        habitat = input_features['habitat']
        season = input_features['season']
        if habitat == 'wetland':
            species = "Blue-winged Teal / Northern Shoveler"
            guild = "Waterfowl & Shorebirds"
        elif habitat == 'forest':
            species = "Wood Thrush / Black-throated Blue Warbler"
            guild = "Neotropical songbirds"
        elif habitat == 'grassland':
            species = "Bobolink / Eastern Meadowlark"
            guild = "Aerial insectivores"
        elif habitat == 'coastal':
            species = "Red Knot / Semipalmated Sandpiper"
            guild = "Shorebirds & Seabirds"
        else:
            species = "American Robin / Common Grackle"
            guild = "Mixed guild / Generalist"
        
        # Adjust for season
        if season == 'winter':
            species = "Dark-eyed Junco / Snow Bunting"
            guild = "Winter residents"
        
        confidence = int(65 + prob * 30)
        
        return {
            'probability': prob_percent,
            'species': species,
            'guild': guild,
            'riskDescription': risk,
            'conservationAdvice': cons,
            'confidence': confidence
        }
    




    