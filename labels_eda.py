import pandas as pd
from pathlib import Path

def labels_eda():
    labels_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/Labels/HECO_Labels.csv")
    df = pd.read_csv(labels_path)
    print(df.head())
    print(df["Category"].unique())
    print(df["Label_SA"].unique())
    print(df["Label_CA"].unique())
    print(df["Valence"].max())
    print(df["Arousal"].max())
    print(df["Dominance"].max())
    print(df["Valence"].min())
    print(df["Arousal"].min())
    print(df["Dominance"].min())
    
    print(df.isna().sum())
    
if __name__ == "__main__":
    labels_eda()