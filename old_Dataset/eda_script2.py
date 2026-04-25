import pandas as pd
import numpy as np

data_path = "d:/College/Programming/Aves_Migration_System/Bird_Migration_Data_with_Origin.csv"
df = pd.read_csv(data_path)

out_text = ""
out_text += "DATA SUMMARY\n" + "-" * 50 + "\n"
out_text += f"Shape: {df.shape}\n"
out_text += f"Columns: {df.columns.tolist()}\n\n"

target_col = "Migration_Success"
if target_col in df.columns:
    df["Target_Numeric"] = df[target_col].map({"Successful": 1, "Failed": 0})
    
    out_text += "CORRELATIONS WITH TARGET (Numeric features)\n" + "-" * 50 + "\n"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Target_Numeric")
    
    correlations = []
    for col in numeric_cols:
        corr = df[col].corr(df["Target_Numeric"])
        correlations.append((col, corr))
        
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations:
        out_text += f"{col:25s}: {corr:.4f}\n"
        
    out_text += "\n\nTARGET DISTRIBUTION\n" + "-" * 50 + "\n"
    out_text += df[target_col].value_counts().to_string() + "\n\n"

with open("d:/College/Programming/Aves_Migration_System/eda_report_utf8.txt", "w", encoding="utf-8") as f:
    f.write(out_text)
