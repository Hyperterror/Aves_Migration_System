import pandas as pd
import numpy as np

# Load the preprocessing logic partly to see what the model actually sees
data_path = "d:/College/Programming/Aves_Migration_System/Bird_Migration_Data_with_Origin.csv"
df = pd.read_csv(data_path)

print("DATA SUMMARY")
print("-" * 50)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create target variable mapping to numeric for correlation
target_col = "Migration_Success"
if target_col in df.columns:
    df["Target_Numeric"] = df[target_col].map({"Successful": 1, "Failed": 0})
    success_rate = df["Target_Numeric"].mean()
    print(f"\nOverall Success Rate: {success_rate:.4f}")

    print("\nCORRELATIONS WITH TARGET (Numeric features)")
    print("-" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Target_Numeric")
    
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(df["Target_Numeric"])
        correlations.append((col, corr))
        
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations:
        print(f"{col:25s}: {corr:.4f}")
        
    print("\nTOP CATEGORICAL DISTRIBUTIONS BY TARGET")
    print("-" * 50)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols.remove(target_col)
    categorical_cols.remove("Origin") # Duplicate
    
    for col in categorical_cols:
        if df[col].nunique() < 10: # Only look at low-cardinality categorical
            print(f"\n--- {col} ---")
            grouped = df.groupby(col)["Target_Numeric"].agg(['mean', 'count'])
            print(grouped)

# We notice that in detailed_report.txt, the model performance was evaluated on 2000 samples.
# The actual full dataset has 10000 samples.
