import numpy as np
from pathlib import Path

def matrices_eda():
    train_path = Path("/content/drive/MyDrive/Affect-Analysis/modeling_matrices")
    eval_path = Path("/content/drive/MyDrive/Affect-Analysis/eval_matrices")
    
    train_matrix = np.load(train_path / "train.npy")
    print(f"train images shape: {train_matrix.shape}")
    del train_matrix
    
    val_matrix = np.load(train_path / "val.npy")
    print(f"val images shape: {val_matrix.shape}")
    del val_matrix
    
    test_matrix = np.load(eval_path / "test.npy")
    print(f"test shape: {test_matrix.shape}")
    del test_matrix
    
    train_labels = np.load(train_path / "train_labels.npy")
    print(f"train labels: {train_labels.shape}")
    
    train_labels = np.load(train_path / "train_labels.npy")
    print(f"train labels: {train_labels.shape}")
    
    train_labels = np.load(train_path / "train_labels.npy")
    print(f"train labels: {train_labels.shape}")
    
if __name__ == "__main__":
    matrices_eda()
