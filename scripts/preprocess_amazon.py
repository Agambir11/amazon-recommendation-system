import pandas as pd
from pathlib import Path

def preprocess_amazon(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load all splits
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    test = pd.read_csv(data_dir / "test.csv")

    full = pd.concat([train, valid, test], ignore_index=True)

    # build mappings
    user_map = {u: i for i, u in enumerate(full["user_id"].unique())}
    item_map = {a: i for i, a in enumerate(full["parent_asin"].unique())}

    def map_split(df):
        return pd.DataFrame({
            "uid": df["user_id"].map(user_map),
            "iid": df["parent_asin"].map(item_map),
            "ts": df["timestamp"]
        })

    train_out = map_split(train)
    valid_out = map_split(valid)
    test_out = map_split(test)

    # save
    train_out.to_csv(out_dir / "train.csv", index=False)
    valid_out.to_csv(out_dir / "valid.csv", index=False)
    test_out.to_csv(out_dir / "test.csv", index=False)

    pd.Series(user_map).to_csv(out_dir / "user_map.csv")
    pd.Series(item_map).to_csv(out_dir / "item_map.csv")

    print(f"✅ Preprocessed: {len(user_map)} users, {len(item_map)} items")
    print(f"Saved cleaned CSVs to {out_dir}")

if __name__ == "__main__":
    preprocess_amazon(
        data_dir="/Users/agambirsingh/Desktop/amazon-recsys/data/amazon23/electronics",
        out_dir="/Users/agambirsingh/Desktop/amazon-recsys/data/amazon23/electronics_clean"
    )
