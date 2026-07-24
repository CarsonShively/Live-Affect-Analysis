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

    train_labels_valence = (train_labels[:, 26:27] - 1.0) / 9.0
    train_labels_arousal = (train_labels[:, 27:28] - 1.0) / 9.0
    val_labels_valence = (val_labels[:, 26:27] - 1.0) / 9.0
    val_labels_arousal = (val_labels[:, 27:28] - 1.0) / 9.0
    
    train_happiness = train_labels[:, 5:6]
    train_calmness = train_labels[:, 9:10]
    train_sadness = train_labels[:, 12:13]
    train_fear = train_labels[:, 16:17]
    train_anger = train_labels[:, 22:23]
    
    val_happiness = val_labels[:, 5:6]
    val_calmness = val_labels[:, 9:10]
    val_sadness = val_labels[:, 12:13]
    val_fear = val_labels[:, 16:17]
    val_anger = val_labels[:, 22:23]
    

    train_happiness = tf.convert_to_tensor(train_happiness, dtype=tf.float32)
    train_calmness = tf.convert_to_tensor(train_calmness, dtype=tf.float32)
    train_sadness = tf.convert_to_tensor(train_sadness, dtype=tf.float32)
    train_fear = tf.convert_to_tensor(train_fear, dtype=tf.float32)
    train_anger = tf.convert_to_tensor(train_anger, dtype=tf.float32)
    train_labels_valence = tf.convert_to_tensor(train_labels_valence, dtype=tf.float32)
    train_labels_arousal = tf.convert_to_tensor(train_labels_arousal, dtype=tf.float32)
    
    
    val_happiness = tf.convert_to_tensor(val_happiness, dtype=tf.float32)
    val_calmness = tf.convert_to_tensor(val_calmness, dtype=tf.float32)
    val_sadness = tf.convert_to_tensor(val_sadness, dtype=tf.float32)
    val_fear = tf.convert_to_tensor(val_fear, dtype=tf.float32)
    val_anger = tf.convert_to_tensor(val_anger, dtype=tf.float32)
    val_labels_valence = tf.convert_to_tensor(val_labels_valence, dtype=tf.float32)
    val_labels_arousal = tf.convert_to_tensor(val_labels_arousal, dtype=tf.float32)


    mean_hapiness = tf.fill(tf.shape(val_happiness), tf.mean(train_happiness))
    mean_calmness = tf.fill(tf.shape(val_calmness), tf.mean(train_calmness))
    mean_sadness = tf.fill(tf.shape(val_sadness), tf.reduce_mean(train_sadness))
    mean_fear = tf.fill(tf.shape(val_fear), tf.mean(train_fear))
    mean_anger = tf.fill(tf.shape(val_anger), tf.mean(train_anger))
    mean_val = tf.fill(tf.shape(val_labels_valence), tf.reduce_mean(train_labels_valence))
    mean_ar = tf.fill(tf.shape(val_labels_arousal), tf.reduce_mean(train_labels_arousal))

    


    
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    
    
    
    happiness_loss = bce(val_happiness, mean_hapiness)
    calmness_loss = bce(val_calmness, mean_calmness)
    sadness_loss = bce(val_sadness, mean_sadness)
    fear_loss = bce(val_fear, mean_fear)
    anger_loss = bce(val_anger, mean_anger)
    valence_loss = mse(val_labels_valence, mean_val)
    arousal_loss = mse(val_labels_arousal, mean_ar)
    
    loss = happiness_loss + calmness_loss + sadness_loss + fear_loss + anger_loss + valence_loss * 10.0 + arousal_loss * 10.0
    
    report = {
        "total_loss": loss.numpy().item(),
        "happiness_bce": happiness_loss.numpy().item(),
        "calmness_bce": calmness_loss.numpy().item(),
        "sadness_bce": sadness_loss.numpy().item(),
        "fear_bce": fear_loss.numpy().item(),
        "anger_bce": anger_loss.numpy().item(),
        "valence_mse": valence_loss.numpy().item(),
        "arousal_mse": arousal_loss.numpy().item()
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