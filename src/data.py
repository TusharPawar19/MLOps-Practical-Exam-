import os
import pandas as pd
from sklearn.datasets import load_digits

os.makedirs("data", exist_ok=True)
data = load_digits()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df.to_csv("data/dataset.csv", index=False)
print("Dataset saved successfully!")
print(df.head())
