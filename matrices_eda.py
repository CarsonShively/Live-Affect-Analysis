import numpy as np
from pathlib import Path
import pandas as pd

def matrices_eda():
    modeling_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    eval_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/eval_matrices")
    
    train_images = np.load(modeling_path / "train_images.npy")
    print(f"train images shape: {train_images.shape}")
    del train_images
    
    val_images = np.load(modeling_path / "val_images.npy")
    print(f"val images shape: {val_images.shape}")
    del val_images
    
    test_images = np.load(eval_path / "test_images.npy")
    print(f"test images shape: {test_images.shape}")
    del test_images
    
    train_labels = np.load(modeling_path / "train_labels.npy")
    print(f"train labels shpae: {train_labels.shape}")
    
    val_labels = np.load(modeling_path / "val_labels.npy")
    print(f"val labels shpae: {val_labels.shape}")
    
    test_labels = np.load(eval_path / "test_labels.npy")
    print(f"train labels shpae: {test_labels.shape}")

    inf_mask = np.isinf(train_labels)
    print(inf_mask.sum())

    train_labels_df = pd.DataFrame(train_labels)
    print(train_labels_df.head())
    print(train_labels_df.isna().sum())

if __name__ == "__main__":
    matrices_eda()