import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob(r"C:\ml-data\cic-ids2017\**\*.csv", recursive=True))
parts = []
for f in files:
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    df["src_file"] = f.split("\\")[-1]
    df["row"] = range(len(df))
    parts.append(df)
df = pd.concat(parts, ignore_index=True)
df["Label"] = df["Label"].str.replace("\u00ef\u00bf\u00bd", "-", regex=False)
feat = [c for c in df.columns if c not in ("Label", "src_file", "row")]
before = df["Label"].value_counts()

drop_cols = [c for c in feat if df[c].nunique() == 1] + ["Fwd Header Length.1"]
df = df.drop(columns=drop_cols)
feat = [c for c in feat if c not in drop_cols]
print("dropped columns:", drop_cols)
print("features left:", len(feat))

bad = ~np.isfinite(df[feat]).all(axis=1)
print("\ninf or nan rows removed, by label")
print(df.loc[bad, "Label"].value_counts().to_string())
df = df[~bad]

h = pd.util.hash_pandas_object(df[feat], index=False)
d = pd.DataFrame({"h": h, "label": df["Label"]})
nl = d.drop_duplicates(["h", "label"]).groupby("h").size()
conf = h.isin(nl[nl > 1].index)
print("\nrows with conflicting labels removed, by label")
print(df.loc[conf, "Label"].value_counts().to_string())
df = df[~conf]

n = len(df)
df = df.drop_duplicates(subset=feat + ["Label"])
print("\nexact duplicates removed:", n - len(df))

df = df.reset_index(drop=True)
after = df["Label"].value_counts()
t = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
t["kept_pct"] = (100 * t["after"] / t["before"]).round(1)
print("\nbefore and after cleaning")
print(t.to_string())
print("rows:", len(df), "of", int(before.sum()))

df.to_parquet(r"C:\ml-data\clean.parquet")
print("saved C:\\ml-data\\clean.parquet")
