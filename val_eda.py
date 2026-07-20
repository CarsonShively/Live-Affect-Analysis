from pathlib import Path
import numpy as np
import pandas as pd

def val_eda():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    
    train_labels = np.load(train_path / "train_labels.npy").astype(np.float32)
    val_labels = np.load(train_path / "val_labels.npy").astype(np.float32)
    
    df_train = pd.DataFrame(train_labels)
    df_val = pd.DataFrame(val_labels)
    
    print(df_train.head())
    print(df_val.head())
    
if __name__ == "__main__":
    val_eda()