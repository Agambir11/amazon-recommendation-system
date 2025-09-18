import pandas as pd

# path to your train.csv
path = "/Users/agambirsingh/Desktop/amazon-recsys/data/amazon23/electronics/train.csv"

# read just first 10 rows
df = pd.read_csv(path, nrows=10)
print(df.head(10))
print("\nColumns:", df.columns.tolist())
