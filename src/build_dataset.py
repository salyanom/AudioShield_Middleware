import pandas as pd

benign = pd.read_csv("features/benign_features.csv")

benign["label"] = 0

benign.to_csv(
    "features/dataset.csv",
    index=False
)

print("Dataset created")
print(benign.head())