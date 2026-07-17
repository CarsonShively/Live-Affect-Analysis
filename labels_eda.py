import pandas as pd
from pathlib import Path

def labels_eda():
    labels_path = Path.home() / "google-drive/Live-Affect-Analysis/Labels/HECO_Labels.csv"
    df = pd.read_csv(labels_path)
    print(df.head())
    
if __name__ == "__main__":
    labels_eda()