import os
import pandas as pd

def save_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
