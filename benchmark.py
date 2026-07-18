from pathlib import Path
import tensorflow as tf
from huggingface_hub import get_token, HfApi
import json
import numpy as np

def benchmark():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    train_labels = np.load(train_path / "train_labels.npy")
    val_labels = np.load(train_path / "val_labels.npy")
    
    train_cat = tf.convert_to_tensor(train_labels[:, 0], dtype=tf.int32)
    train_sat = tf.convert_to_tensor(train_labels[:, 1:2], dtype=tf.float32)
    train_cal = tf.convert_to_tensor(train_labels[:, 2:3], dtype=tf.float32)
    train_val = tf.convert_to_tensor(train_labels[:, 3:4], dtype=tf.float32)
    train_ar = tf.convert_to_tensor(train_labels[:, 4:5], dtype=tf.float32)
    train_dom = tf.convert_to_tensor(train_labels[:, 5:6], dtype=tf.float32)
    

    val_cat = tf.convert_to_tensor(val_labels[:, 0], dtype=tf.int32)
    val_sat = tf.convert_to_tensor(val_labels[:, 1:2], dtype=tf.float32)
    val_cal = tf.convert_to_tensor(val_labels[:, 2:3], dtype=tf.float32)
    val_val = tf.convert_to_tensor(val_labels[:, 3:4], dtype=tf.float32)
    val_ar = tf.convert_to_tensor(val_labels[:, 4:5], dtype=tf.float32)
    val_dom = tf.convert_to_tensor(val_labels[:, 5:6], dtype=tf.float32)


    class_counts = tf.math.bincount(train_cat, dtype=tf.float32)
    
    probabilites = class_counts / tf.reduce_sum(class_counts)
    
    logits = tf.math.log(probabilites + 1e-7)

    logits_matrix = tf.tile(
        logits[tf.newaxis, :],
        multiples=[tf.shape(val_cat)[0], 1]
    )

    mean_sat = tf.fill(tf.shape(val_sat), tf.reduce_mean(train_sat))
    mean_cal = tf.fill(tf.shape(val_cal), tf.reduce_mean(train_cal))
    mean_val = tf.fill(tf.shape(val_val), tf.reduce_mean(train_val))
    mean_ar = tf.fill(tf.shape(val_ar), tf.reduce_mean(train_ar))
    mean_dom = tf.fill(tf.shape(val_dom), tf.reduce_mean(train_dom))
    

    
    
    
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    
    cat_loss = ce(val_cat, logits_matrix)
    sat_loss = mse(val_sat, mean_sat)
    cal_loss = mse(val_cal, mean_cal)
    val_loss = mse(val_val, mean_val)
    ar_loss = mse(val_ar, mean_ar)
    dom_loss = mse(val_dom, mean_dom)
    
    loss = cat_loss + sat_loss + cal_loss + val_loss + ar_loss + dom_loss
    
    report = {
        "total_loss": loss.numpy().item(),
        "category_cross_entropy": cat_loss.numpy().item(),
        "satisfaction_mse": sat_loss.numpy().item(),
        "calmness_mse": cal_loss.numpy().item(),
        "valence_mse": val_loss.numpy().item(),
        "arousal_mse": ar_loss.numpy().item(),
        "dominance": dom_loss.numpy().item()
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