import pandas as pd
from pathlib import Path

def load_data():
    # DF_PATH = Path("..", "assets", "elections.csv")
    DF_PATH = Path("python_libs", "assets", "elections.csv")
    
    df = pd.read_csv(DF_PATH, nrows=1_000)

    return df

def get_num_unique_regions(df):
    return df['marz'].nunique()

def get_first_name(df):
    return df['anun'].iloc[0]

if __name__ == "__main__":
    df = load_data()
    print(f"Number of unique regions: {get_num_unique_regions(df)}")
    print(f"First name in the dataset: {get_first_name(df)}")