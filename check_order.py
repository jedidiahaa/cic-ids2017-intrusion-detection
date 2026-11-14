import pandas as pd

df = pd.read_parquet(r"C:\ml-data\clean.parquet", columns=["src_file", "row", "Label"])
df = df.sort_values(["src_file", "row"]).reset_index(drop=True)
new = (df["Label"] != df["Label"].shift()) | (df["src_file"] != df["src_file"].shift())
df["run"] = new.cumsum()

runs = df.groupby(["Label", "run"]).size().reset_index(name="n")
out = runs.groupby("Label")["n"].agg(runs="count", median_len="median", longest="max", rows="sum")
print("contiguous runs of each label in file order")
print(out.to_string())

df["pos"] = (df["row"] / df.groupby("src_file")["row"].transform("max")).round(2)
a = df[df["Label"] != "BENIGN"].groupby(["src_file", "Label"])["pos"].agg(["min", "max", "count"])
print("\nwhere attacks sit inside each file (0 = start, 1 = end)")
print(a.to_string())
