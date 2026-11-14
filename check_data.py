import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob(r"C:\ml-data\cic-ids2017\**\*.csv", recursive=True))
parts = []
for f in files:
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    df["src_file"] = f.split("\\")[-1]
    parts.append(df)
df = pd.concat(parts, ignore_index=True)
df["Label"] = df["Label"].str.replace("\u00ef\u00bf\u00bd", "-", regex=False)
feat = [c for c in df.columns if c not in ("Label", "src_file")]
cols = feat + ["Label"]

print("labels:", sorted(df["Label"].unique()))

num = df[feat]
bad = pd.DataFrame({"inf": np.isinf(num).sum(), "nan": num.isna().sum()})
print("\ncolumns with inf or nan")
print(bad[bad.sum(axis=1) > 0].to_string())
print("\nconstant columns:", [c for c in feat if df[c].nunique() == 1])
print("Fwd Header Length twin equal:", bool((df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()))

dup = df.duplicated(subset=cols)
t = pd.DataFrame({"rows": df["Label"].value_counts(), "dups": df.loc[dup, "Label"].value_counts()}).fillna(0).astype(int)
t["dup_pct"] = (100 * t["dups"] / t["rows"]).round(1)
print("\nexact duplicates by label")
print(t.to_string())

h = pd.util.hash_pandas_object(df[feat], index=False)
d = pd.DataFrame({"h": h, "label": df["Label"], "file": df["src_file"]})
nl = d.drop_duplicates(["h", "label"]).groupby("h").size()
nf = d.drop_duplicates(["h", "file"]).groupby("h").size()
print("\ndistinct feature rows:", len(nl))
print("feature rows with more than one label:", int((nl > 1).sum()))
print("feature rows seen in more than one file:", int((nf > 1).sum()))
