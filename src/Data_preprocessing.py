import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    le = LabelEncoder()
    df["department"] = le.fit_transform(df["department"])
    df["performance"] = le.fit_transform(df["performance"])
    return df