import tensorflow as tf
from pathlib import Path
import numpy as np

def train_model():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    
    train_images = np.load(train_path / "train_images.npy")
    train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.uint8)
    del train_images
    
    val_images = np.load(train_path / "val_images.npy")
    val_images_tensor = tf.convert_to_tensor(val_images, dtype=tf.uint8)
    del val_images
    
    train_labels = np.load(train_path / "train_labels.npy")
    train_labels_tensor = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    del train_labels
    
    val_labels = np.load(train_path / "val_labels.npy")
    val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)
    del val_labels
    
    print(f"train images shape: {train_images_tensor.shape}")
    print(f"val images shape: {val_images_tensor.shape}")
    print(f"train labels shape: {train_labels_tensor.shape}")
    print(f"val labels shape: {val_labels_tensor.shape}")
    
if __name__ == "__main__":
    train_model()