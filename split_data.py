import numpy as np
import pandas as pd

bs = 1000
df = pd.read_parquet(r"C:\ml-data\clean.parquet")
rng = np.random.default_rng(42)

df["block"] = df["src_file"] + "_" + (df["row"] // bs).astype(str)
blocks = df["block"].unique()
test_blocks = rng.choice(blocks, size=int(0.3 * len(blocks)), replace=False)
df["split_block"] = np.where(df["block"].isin(test_blocks), "test", "train")
df["split_rand"] = np.where(rng.random(len(df)) < 0.3, "test", "train")
print("blocks:", len(blocks), "test blocks:", len(test_blocks))

for s in ["split_block", "split_rand"]:
    t = pd.crosstab(df["Label"], df[s])
    t["test_pct"] = (100 * t["test"] / (t["train"] + t["test"])).round(1)
    print("\n", s)
    print(t.to_string())

df.to_parquet(r"C:\ml-data\split.parquet")
print("\nsaved C:\\ml-data\\split.parquet")
