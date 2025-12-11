import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def split(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
