"""
Telemetry Bucket Splitter
==========================
Companion to bucket_splitter.py — works with telemetry-derived features
instead of the pre-aggregated Bird_Migration_Data_with_Origin.csv.

Usage
-----
    # Create all bucket CSVs from the aggregated telemetry data:
    from telemetry_bucket_splitter import create_telemetry_buckets
    create_telemetry_buckets()

    # Or access individual bucket DataFrames directly:
    from telemetry_bucket_splitter import TelemetryBucketSplitter
    splitter = TelemetryBucketSplitter()
    train_df, test_df = splitter.get_bucket("MOVEMENT")

Buckets
-------
    GEOSPATIAL  – Start/end coordinates, total distance, displacement,
                  route tortuosity, max distance from breeding, 
                  bearing statistics.

    TEMPORAL    – Migration duration, start/end month, day-of-year,
                  average daily distance, mode movement hour.

    MOVEMENT    – Average/max/std speed, heading consistency, GPS fix
                  count and fix rate per day.

    SENSOR      – Temperature statistics (avg, std, min, max), battery
                  voltage statistics, depletion rate, horizontal
                  accuracy, time-to-fix.

    ALTITUDE    – Average, max, min, range, and std of altitude above
                  ellipsoid throughout the migration event.

    BEHAVIOUR   – Rest stop count and duration, movement consistency
                  ratio, number of long flight segments, max consecutive
                  movement hours.

    FULL        – All features combined (baseline bucket for full-model
                  training, mirrors bucket_splitter.py FULL bucket).

Why these buckets?
------------------
Each bucket represents a distinct facet of migration biology:
* GEOSPATIAL  → where the bird travelled and how efficiently
* TEMPORAL    → when and for how long it migrated
* MOVEMENT    → fine-grained locomotion characteristics
* SENSOR      → physiological and tag-health proxies
* ALTITUDE    → flight envelope and energy expenditure
* BEHAVIOUR   → rest and activity patterns (fatigue indicators)
Splitting into buckets allows teams to analyse each domain independently
and then combine the best features for the FULL model.
"""

import os
import pandas as pd
from telemetry_preprocess import preprocess_telemetry, OUTPUT_CSV, DATA_PATH

# ================================================================
# BUCKET DEFINITIONS (mirrors structure in bucket_splitter.py)
# ================================================================

TELEMETRY_BUCKET_DEFINITIONS = {

    "GEOSPATIAL": {
        "description": (
            "Spatial route metrics — start/end coordinates, total path "
            "length, displacement from origin, route tortuosity, maximum "
            "distance from breeding colony, and bearing statistics."
        ),
        "columns": [
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "total_distance_km",
            "displacement_km",
            "route_tortuosity",
            "max_dist_from_breeding_km",
            "bearing_mean",
            "bearing_std",
        ],
    },

    "TEMPORAL": {
        "description": (
            "Timing and duration metrics — migration duration in days, "
            "calendar start/end months, day-of-year for departure/arrival, "
            "average daily distance, and modal hour-of-day for movement "
            "(indicates diurnal vs. nocturnal migration strategy)."
        ),
        "columns": [
            "duration_days",
            "start_month",
            "end_month",
            "start_doy",
            "end_doy",
            "avg_daily_distance_km",
            "movement_hour_mode",
        ],
    },

    "MOVEMENT": {
        "description": (
            "Locomotion metrics derived from GPS ground-speed and heading — "
            "average / maximum / std of speed, heading consistency index "
            "(mean resultant length, 0–1), total GPS fix count, and fix "
            "rate per day (data density indicator)."
        ),
        "columns": [
            "avg_speed_ms",
            "max_speed_ms",
            "speed_std",
            "heading_consistency",
            "n_gps_fixes",
            "fix_rate_per_day",
        ],
    },

    "SENSOR": {
        "description": (
            "Physiological and tag-health proxies from onboard sensors — "
            "temperature statistics (avg, std, min, max), battery voltage "
            "stats (avg, min), battery depletion rate (V/day), average "
            "horizontal accuracy, and average time-to-first-fix."
        ),
        "columns": [
            "avg_temperature",
            "temp_std",
            "min_temperature",
            "max_temperature",
            "avg_battery_v",
            "min_battery_v",
            "battery_depletion_rate",
            "avg_horiz_accuracy",
            "avg_time_to_fix",
        ],
    },

    "ALTITUDE": {
        "description": (
            "Altitude profile metrics from height-above-ellipsoid — "
            "average, maximum, minimum, range (max−min), and standard "
            "deviation of altitude throughout the migration event."
        ),
        "columns": [
            "avg_altitude_m",
            "max_altitude_m",
            "min_altitude_m",
            "altitude_range_m",
            "altitude_std",
        ],
    },

    "BEHAVIOUR": {
        "description": (
            "Behavioural pattern features — number of rest stops detected, "
            "average rest stop duration (hours), movement consistency ratio "
            "(fraction of fixes with speed > 0.5 m/s), count of long "
            "flight segments (≥2 hr continuous movement), and maximum "
            "consecutive movement hours in a single flight."
        ),
        "columns": [
            "n_rest_stops",
            "avg_rest_duration_hrs",
            "movement_consistency_ratio",
            "long_flight_segments",
            "max_consecutive_movement_hrs",
        ],
    },

    "FULL": {
        "description": (
            "All telemetry-derived features combined. "
            "Use this bucket to train the baseline full-feature model, "
            "mirroring the FULL bucket in bucket_splitter.py."
        ),
        "columns": None,   # will be set to all feature columns at runtime
    },
}


# ================================================================
# BUCKET SPLITTER CLASS (mirrors BucketSplitter in bucket_splitter.py)
# ================================================================

class TelemetryBucketSplitter:
    """
    Loads preprocessed telemetry migration data and provides
    per-bucket train/test DataFrames.

    Mirrors the interface of BucketSplitter in bucket_splitter.py
    so downstream code can switch between the two datasets easily.
    """

    def __init__(
        self,
        aggregated_csv: str = OUTPUT_CSV,
        data_path: str = DATA_PATH,
        test_size: float = 0.2,
        random_state: int = 42,
        rebuild: bool = False,
    ):
        """
        Parameters
        ----------
        aggregated_csv : str
            Path to the pre-aggregated migration events CSV.
            If missing, the telemetry pipeline is run automatically.
        data_path : str
            Path to the raw telemetry CSV (used only if rebuilding).
        test_size : float
            Fraction of events reserved for the test set.
        random_state : int
            Reproducibility seed.
        rebuild : bool
            Force re-running the aggregation pipeline.
        """
        print("Loading preprocessed telemetry data …")
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = preprocess_telemetry(
            aggregated_csv=aggregated_csv,
            test_size=test_size,
            random_state=random_state,
            rebuild=rebuild,
            data_path=data_path,
        )

        # Register all feature columns in the FULL bucket at runtime
        if TELEMETRY_BUCKET_DEFINITIONS["FULL"]["columns"] is None:
            TELEMETRY_BUCKET_DEFINITIONS["FULL"]["columns"] = list(
                self.X_train.columns
            )

    # ----------------------------------------------------------------
    def get_bucket(self, bucket_name: str, include_label: bool = True):
        """
        Return (train_df, test_df) for the requested bucket.

        Parameters
        ----------
        bucket_name : str
            One of: GEOSPATIAL, TEMPORAL, MOVEMENT, SENSOR,
            ALTITUDE, BEHAVIOUR, FULL
        include_label : bool
            Append Migration_Success column when True.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame] : (train_df, test_df)
        """
        if bucket_name not in TELEMETRY_BUCKET_DEFINITIONS:
            valid = list(TELEMETRY_BUCKET_DEFINITIONS.keys())
            raise ValueError(
                f"Unknown bucket: '{bucket_name}'. Valid options: {valid}"
            )

        bucket_cols = TELEMETRY_BUCKET_DEFINITIONS[bucket_name]["columns"]

        if bucket_cols is None:
            # FULL bucket: use everything
            X_train_b = self.X_train.copy()
            X_test_b  = self.X_test.copy()
        else:
            valid_cols = [c for c in bucket_cols if c in self.X_train.columns]
            missing = [c for c in bucket_cols if c not in self.X_train.columns]
            if missing:
                print(f"  [WARN] {bucket_name}: missing columns {missing}")
            X_train_b = self.X_train[valid_cols].copy()
            X_test_b  = self.X_test[valid_cols].copy()

        if include_label:
            X_train_b = X_train_b.copy()
            X_test_b  = X_test_b.copy()
            X_train_b["Migration_Success"] = self.y_train.values
            X_test_b["Migration_Success"]  = self.y_test.values

        return X_train_b, X_test_b

    # ----------------------------------------------------------------
    def export_bucket(self, bucket_name: str, output_dir: str = "telemetry_bucket_data"):
        """
        Write train/test CSVs for one bucket to disk.

        Parameters
        ----------
        bucket_name : str
        output_dir  : str   Directory to write files into.
        """
        os.makedirs(output_dir, exist_ok=True)

        train_df, test_df = self.get_bucket(bucket_name)

        train_path = os.path.join(output_dir, f"{bucket_name.lower()}_train.csv")
        test_path  = os.path.join(output_dir, f"{bucket_name.lower()}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        print(f"  Exported {bucket_name}:")
        print(f"    {train_path}  ({len(train_df)} rows, {len(train_df.columns)} cols)")
        print(f"    {test_path}   ({len(test_df)} rows, {len(test_df.columns)} cols)")

    # ----------------------------------------------------------------
    def export_all_buckets(self, output_dir: str = "telemetry_bucket_data"):
        """Export all buckets to the output directory."""
        print(f"\nExporting all telemetry buckets to '{output_dir}' …")
        for bucket_name in TELEMETRY_BUCKET_DEFINITIONS:
            self.export_bucket(bucket_name, output_dir)
        print("\nAll buckets exported.")

    # ----------------------------------------------------------------
    def print_summary(self):
        """Print a summary table of all buckets."""
        print("\n" + "=" * 65)
        print("TELEMETRY BUCKET SUMMARY")
        print("=" * 65)
        print(f"  Train samples : {len(self.X_train)}")
        print(f"  Test samples  : {len(self.X_test)}")
        print(f"  Total features: {len(self.X_train.columns)}")
        print()

        header = f"  {'Bucket':<15}  {'# Cols':>6}  Description"
        print(header)
        print("  " + "-" * 62)

        for name, defn in TELEMETRY_BUCKET_DEFINITIONS.items():
            if name == "FULL":
                n_cols = len(self.X_train.columns)
            else:
                valid = [c for c in defn["columns"] if c in self.X_train.columns]
                n_cols = len(valid)
            desc = defn["description"].split("—")[0].strip()
            print(f"  {name:<15}  {n_cols:>6}  {desc}")

        print()


# ================================================================
# CONVENIENCE FUNCTION (mirrors create_buckets() in bucket_splitter.py)
# ================================================================

def create_telemetry_buckets(
    output_dir: str = "telemetry_bucket_data",
    aggregated_csv: str = OUTPUT_CSV,
    data_path: str = DATA_PATH,
    rebuild: bool = False,
) -> "TelemetryBucketSplitter":
    """
    One-stop function: load data, print summary, export all bucket CSVs.

    Parameters
    ----------
    output_dir     : str   Where to write CSVs.
    aggregated_csv : str   Pre-aggregated migration events file.
    data_path      : str   Raw telemetry CSV (only used if rebuilding).
    rebuild        : bool  Re-run aggregation pipeline if True.

    Returns
    -------
    TelemetryBucketSplitter
    """
    splitter = TelemetryBucketSplitter(
        aggregated_csv=aggregated_csv,
        data_path=data_path,
        rebuild=rebuild,
    )
    splitter.print_summary()
    splitter.export_all_buckets(output_dir)
    return splitter


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    create_telemetry_buckets()
