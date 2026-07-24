from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"
import tensorflow as tf
from huggingface_hub import get_token, HfApi
import json
import numpy as np

def benchmark():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    train_labels = np.load(train_path / "train_labels.npy")
    val_labels = np.load(train_path / "val_labels.npy")

    
    train_category = tf.convert_to_tensor(train_labels[:, :26], dtype=tf.float32)
    train_valence = tf.convert_to_tensor((train_labels[:, 26:27] - 1) / 9, dtype=tf.float32)
    train_arousal = tf.convert_to_tensor((train_labels[:, 27:28] - 1) / 9, dtype=tf.float32)

    val_category = tf.convert_to_tensor(val_labels[:, :26], dtype=tf.float32)
    val_valence = tf.convert_to_tensor((val_labels[:, 26:27] - 1) / 9, dtype=tf.float32)
    val_arousal = tf.convert_to_tensor((val_labels[:, 27:28] - 1) / 9, dtype=tf.float32)

    category_probabilities = tf.reduce_mean(train_category, axis=0)
    mean_categories = tf.broadcast_to(category_probabilities, val_category.shape)



    mean_val = tf.fill(tf.shape(val_valence), tf.reduce_mean(train_valence))
    mean_ar = tf.fill(tf.shape(val_arousal), tf.reduce_mean(train_arousal))

    


    
    
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    
    
    
    cat_loss = bce(val_category, mean_categories)
    val_loss = mse(val_valence, mean_val)
    ar_loss = mse(val_arousal, mean_ar)
    
    loss = cat_loss + val_loss * 10.0 + ar_loss * 10.0
    
    report = {
        "total_loss": loss.numpy().item(),
        "category_bce": cat_loss.numpy().item(),
        "valence_mse": val_loss.numpy().item(),
        "arousal_mse": ar_loss.numpy().item()
    }
    
    out_path = Path("/content/benchmark.json")
    
    with open(out_path, "w") as con:
        json.dump(report, con)
        
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            path_or_fileobj=out_path,
            path_in_repo="benchmark.json"
        )
        
if __name__ == "__main__":
    benchmark()