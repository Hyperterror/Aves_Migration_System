"""
Bucket Data Splitter for Bird Migration Dataset

This module divides preprocessed data into semantic buckets for team collaboration.
Each bucket contains a specific feature subset with labels.

Usage:
    from bucket_splitter import create_buckets
    
    # Create all bucket CSVs
    create_buckets()
    
    # Or access individual bucket DataFrames
    from bucket_splitter import BucketSplitter
    splitter = BucketSplitter()
    
    behaviour_df = splitter.get_bucket("BEHAVIOUR")
    weather_df = splitter.get_bucket("WEATHER")
    # etc.
"""

import os
import pandas as pd
from data_preprocess import preprocess_data


BUCKET_DEFINITIONS = {
    "BEHAVIOUR": {
        "description": "Bird behavior and social/migration patterns",
        "columns": [
            "Nesting_Success_1",
            "Migrated_in_Flock_1",
            "Flock_Size",
            "Rest_Stops",
            "Predator_Sightings",
            "Food_Supply_Level_Low",
            "Food_Supply_Level_Medium",
            "Migration_Interrupted_1",
            "Interrupted_Reason_Lost Signal",
            "Interrupted_Reason_None",
            "Interrupted_Reason_Predator",
            "Interrupted_Reason_Storm",
            "Interrupted_Reason_Unknown",
        ]
    },
    
    "WEATHER": {
        "description": "Environmental and weather conditions during migration",
        "columns": [
            "Temperature_C",
            "Wind_Speed_kmph",
            "Humidity_%",
            "Pressure_hPa",
            "Visibility_km",
            "Weather_Condition_Foggy",
            "Weather_Condition_Rainy",
            "Weather_Condition_Stormy",
            "Weather_Condition_Windy",
        ]
    },
    
    "GEOSPATIAL": {
        "description": "Location, route, and distance information",
        "columns": [
            "Start_Latitude",
            "Start_Longitude",
            "End_Latitude",
            "End_Longitude",
            "Flight_Distance_km",
            "Region_Asia",
            "Region_Australia",
            "Region_Europe",
            "Region_North America",
            "Region_South America",
            "Habitat_Forest",
            "Habitat_Grassland",
            "Habitat_Mountain",
            "Habitat_Urban",
            "Habitat_Wetland",
        ]
    },
    
    "TEMPORAL": {
        "description": "Time-related features (migration timing)",
        "columns": [
            "Migration_Start_Month",
            "Migration_End_Month",
        ]
    },
    
    "TAG": {
        "description": "Tag/device and tracking related features",
        "columns": [
            "Tag_Battery_Level_%",
            "Signal_Strength_dB",
            "Tag_Type_Radio",
            "Tag_Type_Satellite",
            "Tag_Weight_g",
            "Tracking_Quality_Fair",
            "Tracking_Quality_Good",
            "Tracking_Quality_Poor",
        ]
    },
    
    "PHYSIOLOGICAL": {
        "description": "Bird species and physical characteristics",
        "columns": [
            "Species_Eagle",
            "Species_Goose",
            "Species_Hawk",
            "Species_Stork",
            "Species_Swallow",
            "Species_Warbler",
            "Average_Speed_kmph",
            "Max_Altitude_m",
            "Min_Altitude_m",
            "Flight_Duration_hours",
        ]
    },
    
    "FULL": {
        "description": "All features combined",
        "columns": None
    }
}


class BucketSplitter:
    """
    Loads preprocessed data and provides bucket access methods.
    """
    
    def __init__(self, data_path="Bird_Migration_Data_with_Origin.csv"):
        print("Loading preprocessed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(data_path)
        
        if BUCKET_DEFINITIONS["FULL"]["columns"] is None:
            BUCKET_DEFINITIONS["FULL"]["columns"] = list(self.X_train.columns)
    
    def get_bucket(self, bucket_name, include_label=True):
        """
        Get train/test DataFrames for a specific bucket.
        
        Parameters
        ----------
        bucket_name : str
            BEHAVIOUR, WEATHER, GEOSPATIAL, TEMPORAL, TAG, PHYSIOLOGICAL, FULL
        include_label : bool
            If True, includes Migration_Success column
            
        Returns
        -------
        tuple : (train_df, test_df)
        """
        if bucket_name not in BUCKET_DEFINITIONS:
            raise ValueError(f"Unknown bucket: {bucket_name}")
        
        bucket_cols = BUCKET_DEFINITIONS[bucket_name]["columns"]
        
        if bucket_cols is None:
            X_train_bucket = self.X_train.copy()
            X_test_bucket = self.X_test.copy()
        else:
            valid_cols = [col for col in bucket_cols if col in self.X_train.columns]
            X_train_bucket = self.X_train[valid_cols].copy()
            X_test_bucket = self.X_test[valid_cols].copy()
        
        if include_label:
            X_train_bucket = X_train_bucket.copy()
            X_train_bucket["Migration_Success"] = self.y_train.values
            
            X_test_bucket = X_test_bucket.copy()
            X_test_bucket["Migration_Success"] = self.y_test.values
        
        return X_train_bucket, X_test_bucket
    
    def export_bucket(self, bucket_name, output_dir="bucket_data"):
        """
        Export a bucket as train/test CSV files.
        
        Parameters
        ----------
        bucket_name : str
            Bucket to export
        output_dir : str
            Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train_df, test_df = self.get_bucket(bucket_name)
        
        train_path = f"{output_dir}/{bucket_name.lower()}_train.csv"
        test_path = f"{output_dir}/{bucket_name.lower()}_test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Exported {bucket_name}:")
        print(f"  {train_path} ({len(train_df)} rows, {len(train_df.columns)} cols)")
        print(f"  {test_path} ({len(test_df)} rows, {len(test_df.columns)} cols)")
    
    def export_all_buckets(self, output_dir="bucket_data"):
        """Export all buckets to CSV files."""
        print("\nExporting all buckets...")
        for bucket_name in BUCKET_DEFINITIONS:
            self.export_bucket(bucket_name, output_dir)
        print("\nDone!")
    
    def print_summary(self):
        """Print summary of all buckets."""
        print("\n" + "="*60)
        print("BUCKET SUMMARY")
        print("="*60)
        
        for bucket_name, definition in BUCKET_DEFINITIONS.items():
            if bucket_name == "FULL":
                n_cols = len(self.X_train.columns)
            else:
                valid = [c for c in definition["columns"] if c in self.X_train.columns]
                n_cols = len(valid)
            
            print(f"\n{bucket_name} ({n_cols} features)")
            print(f"  {definition['description']}")


def create_buckets(output_dir="bucket_data"):
    """
    Convenience function to create all bucket CSV files.
    
    Parameters
    ----------
    output_dir : str
        Directory to save CSV files
    """
    splitter = BucketSplitter()
    splitter.print_summary()
    splitter.export_all_buckets(output_dir)
    return splitter


if __name__ == "__main__":
    create_buckets()
