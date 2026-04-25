"""
Bird Migration Data Preprocessing Pipeline

This module handles all data preprocessing steps for the bird migration prediction model:
- Data loading and exploration
- Missing value imputation
- Feature encoding (binary, ordinal, one-hot)
- Train/test splitting with stratification
- Feature scaling

Why each preprocessing step is needed:
1. Drop identifiers: Bird_ID and Origin are unique identifiers, not predictive features
2. Handle missing values: ML models cannot handle NaN values
3. Encode categoricals: ML models require numerical input
4. Scale numeric features: Many ML algorithms (especially distance/gradient-based) perform better 
   when features are on similar scales
5. Train/test split: Always test on unseen data to evaluate generalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data_path="Bird_Migration_Data_with_Origin.csv", test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for bird migration data.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV data file
    test_size : float
        Proportion of data to use for testing (default 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features
    X_test_scaled : pd.DataFrame
        Scaled test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    """
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    df = pd.read_csv(data_path)
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(3))
    print("\nData types:")
    print(df.dtypes)
    
    # ============================================================
    # STEP 2: Drop Unnecessary Columns
    # ============================================================
    # Bird_ID: Pure identifier, no predictive value
    # Origin: Duplicate information already present in Start_Latitude/Start_Longitude
    cols_to_drop = ["Bird_ID", "Origin"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    # ============================================================
    # STEP 3: Identify Column Types
    # ============================================================
    # Numeric columns: Continuous or discrete numerical values
    # Categorical columns: Text-based categories that need encoding
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # ============================================================
    # STEP 4: Check Missing Values
    # ============================================================
    missing_counts = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values per column:")
    print(missing_counts[missing_counts > 0])
    
    # ============================================================
    # STEP 5: Handle Missing Values
    # ============================================================
    
    # 5a) Categorical: Fill with "Unknown" as a valid category
    # This is preferred over dropping rows because:
    # - Preserves data quantity for learning other patterns
    # - "Unknown" is a meaningful category (e.g., reason not recorded)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    
    # 5b) Special case: Interrupted_Reason
    # If Migration_Interrupted is "No", the reason should logically be "None"
    # This prevents spurious correlations between reason and interruption status
    if "Migration_Interrupted" in df.columns and "Interrupted_Reason" in df.columns:
        # Convert to string first to handle potential type mismatches
        df["Migration_Interrupted"] = df["Migration_Interrupted"].astype(str)
        df.loc[df["Migration_Interrupted"] == "No", "Interrupted_Reason"] = "None"
        df["Interrupted_Reason"] = df["Interrupted_Reason"].fillna("Unknown")
    
    # 5c) Numeric: Fill with median
    # Median is preferred over mean because:
    # - Robust to outliers (mean can be skewed by extreme values)
    # - Represents the "typical" value better for skewed distributions
    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    # ============================================================
    # STEP 6: Handle Infinite Values
    # ============================================================
    # Replace inf/-inf with NaN, then fill with median
    # This can occur from division operations or corrupted data
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # ============================================================
    # STEP 7: Encode Binary Yes/No Columns
    # ============================================================
    # These columns have only two values: Yes/No
    # Binary encoding (1/0) is more efficient than one-hot for binary variables
    yes_no_cols = [
        "Nesting_Success",
        "Migrated_in_Flock",
        "Migration_Interrupted",
        "Recovery_Location_Known"
    ]
    
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    
    # ============================================================
    # STEP 8: Encode Ordinal Features (Months)
    # ============================================================
    # Migration_Start_Month and Migration_End_Month have NATURAL ORDERING
    # Jan < Feb < Mar ... < Dec
    # 
    # Why ordinal encoding instead of one-hot?
    # - Preserves the ordinal relationship (Dec - Jan = 11, Jan - Feb = 1)
    # - Reduces dimensionality (1 column vs 12 columns)
    # - Many ML algorithms can leverage this ordering
    month_order = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    
    if "Migration_Start_Month" in df.columns:
        df["Migration_Start_Month"] = df["Migration_Start_Month"].map(month_order)
    if "Migration_End_Month" in df.columns:
        df["Migration_End_Month"] = df["Migration_End_Month"].map(month_order)
    
    # ============================================================
    # STEP 9: Encode Target Variable
    # ============================================================
    # Migration_Success is our prediction target
    target_col = "Migration_Success"
    
    # Validate target values before encoding
    unique_targets = df[target_col].unique()
    expected_values = {"Successful", "Failed"}
    unexpected = set(unique_targets) - expected_values
    if unexpected:
        print(f"Warning: Unexpected target values found: {unexpected}")
    
    df[target_col] = df[target_col].map({
        "Successful": 1,
        "Failed": 0
    })
    
    # ============================================================
    # STEP 10: One-Hot Encode Remaining Categorical Features
    # ============================================================
    # Remaining categorical columns (Species, Region, Habitat, etc.)
    # use one-hot encoding because:
    # - No natural ordering exists (Warbler != Sparrow)
    # - Each category is independent (nominal, not ordinal)
    # 
    # drop_first=True: Avoids multicollinearity (dummy variable trap)
    # Example: For Region with 3 values, only 2 columns are needed
    # because the 3rd is fully determined by (not A) AND (not B)
    
    cat_features = [col for col in categorical_cols if col != target_col]
    
    # Remove months from one-hot encoding (already encoded above)
    cat_features = [col for col in cat_features 
                    if col not in ["Migration_Start_Month", "Migration_End_Month"]]
    
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
    
    print(f"\nAfter encoding: {df_encoded.shape[1]} columns")
    
    # ============================================================
    # STEP 11: Split into Features and Target
    # ============================================================
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # ============================================================
    # STEP 12: Train-Test Split with Stratification
    # ============================================================
    # stratify=y: Maintains the same class distribution in train and test sets
    # This is CRITICAL for imbalanced datasets (e.g., 95% success, 5% failure)
    # Without stratification, test set might have different class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class distribution in train: {y_train.value_counts().to_dict()}")
    
    # ============================================================
    # STEP 13: Feature Scaling (Standardization)
    # ============================================================
    # StandardScaler transforms features to have mean=0 and std=1
    # Formula: z = (x - mean) / std
    #
    # Why scale?
    # 1. Gradient-based optimizers converge faster
    # 2. Distance-based algorithms (KNN, SVM) are distance-sensitive
    # 3. Regularization terms treat features equally
    #
    # Why fit ONLY on training data?
    # - Prevents data leakage from test set
    # - Test data should be "unseen" and treated as truly new data
    # - Fitting on all data would give the model an unfair advantage
    
    numeric_cols_final = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    
    # Create new DataFrames with scaled values
    # Using .copy() ensures we don't modify original DataFrames
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # fit_transform: Compute parameters (mean, std) on train, then transform
    # transform: Use the train-computed parameters to transform test
    X_train_scaled[numeric_cols_final] = scaler.fit_transform(X_train[numeric_cols_final])
    X_test_scaled[numeric_cols_final] = scaler.transform(X_test[numeric_cols_final])
    
    print(f"\nScaled {len(numeric_cols_final)} numeric features")
    print(f"Train mean range: [{X_train_scaled[numeric_cols_final].mean().min():.4f}, "
          f"{X_train_scaled[numeric_cols_final].mean().max():.4f}]")
    print(f"Train std range: [{X_train_scaled[numeric_cols_final].std().min():.4f}, "
          f"{X_train_scaled[numeric_cols_final].std().max():.4f}]")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
