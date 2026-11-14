import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob(r"C:\ml-data\cic-ids2017\**\*.csv", recursive=True))
print(len(files), "csv files found")
parts = []
for f in files:
    name = f.split("\\")[-1]
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    df["src_file"] = name
    print("\n==", name, df.shape)
    print(df["Label"].value_counts().to_string())
    parts.append(df)

df = pd.concat(parts, ignore_index=True)
cols = [c for c in df.columns if c != "src_file"]

print("\n== all files")
print("shape:", df.shape)
print("memory GB:", round(df.memory_usage(deep=True).sum() / 1e9, 2))
print("columns:", list(df.columns))
print("dtypes:", df.dtypes.value_counts().to_dict())
print(df["Label"].value_counts().to_string())
num = df.select_dtypes("number")
print("inf values:", int(np.isinf(num).sum().sum()))
print("nan values:", int(df[cols].isna().sum().sum()))
print("rows with inf or nan:", int((~np.isfinite(num)).any(axis=1).sum()))
print("duplicate rows:", int(df.duplicated(subset=cols).sum()))
