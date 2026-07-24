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
    
    train_confusion = train_labels[:, 1]
    train_happiness = np.max(train_labels[:, [5, 7, 4]], axis=1, keepdims=True)
    train_connfidence = np.max(train_labels[:, [8, 19, 3]], axis=1, keepdims=True)
    train_calmness = train_labels[:, 9]
    train_distress = np.max(train_labels[:, [11, 14, 12, 18, 24]], axis=1, keepdims=True)
    train_fear = np.max(train_labels[:, [16, 25]], axis=1, keepdims=True)
    train_anger = np.max(train_labels[:, [22, 20, 6, 23]], axis=1, keepdims=True)
    
    val_confusion = val_labels[:, 1]
    val_happiness = np.max(val_labels[:, [5, 7, 4]], axis=1, keepdims=True)
    val_connfidence = np.max(val_labels[:, [8, 19, 3]], axis=1, keepdims=True)
    val_calmness = val_labels[:, 9]
    val_distress = np.max(val_labels[:, [11, 14, 12, 18, 24]], axis=1, keepdims=True)
    val_fear = np.max(val_labels[:, [16, 25]], axis=1, keepdims=True)
    val_anger = np.max(val_labels[:, [22, 20, 6, 23]], axis=1, keepdims=True)
    

    train_confusion = tf.convert_to_tensor(train_confusion, dtype=tf.float32)
    train_happiness = tf.convert_to_tensor(train_happiness, dtype=tf.float32)
    train_connfidence = tf.convert_to_tensor(train_connfidence, dtype=tf.float32)
    train_calmness = tf.convert_to_tensor(train_calmness, dtype=tf.float32)
    train_distress = tf.convert_to_tensor(train_distress, dtype=tf.float32)
    train_fear = tf.convert_to_tensor(train_fear, dtype=tf.float32)
    train_anger = tf.convert_to_tensor(train_anger, dtype=tf.float32)
    train_labels_valence = tf.convert_to_tensor(train_labels_valence, dtype=tf.float32)
    train_labels_arousal = tf.convert_to_tensor(train_labels_arousal, dtype=tf.float32)
    
    
    val_confusion = tf.convert_to_tensor(val_confusion, dtype=tf.float32)
    val_happiness = tf.convert_to_tensor(val_happiness, dtype=tf.float32)
    val_connfidence = tf.convert_to_tensor(val_connfidence, dtype=tf.float32)
    val_calmness = tf.convert_to_tensor(val_calmness, dtype=tf.float32)
    val_distress = tf.convert_to_tensor(val_distress, dtype=tf.float32)
    val_fear = tf.convert_to_tensor(val_fear, dtype=tf.float32)
    val_anger = tf.convert_to_tensor(val_anger, dtype=tf.float32)
    val_labels_valence = tf.convert_to_tensor(val_labels_valence, dtype=tf.float32)
    val_labels_arousal = tf.convert_to_tensor(val_labels_arousal, dtype=tf.float32)


    mean_confusion = tf.fill(tf.shape(val_confusion), tf.mean(train_confusion))
    mean_hapiness = tf.fill(tf.shape(val_happiness), tf.mean(train_happiness))
    mean_confidence = tf.fill(tf.shape(val_connfidence), tf.mean(train_connfidence))
    mean_calmness = tf.fill(tf.shape(val_calmness), tf.mean(train_calmness))
    mean_distress = tf.fill(tf.shape(val_distress), tf.reduce_mean(train_distress))
    mean_fear = tf.fill(tf.shape(val_fear), tf.mean(train_fear))
    mean_anger = tf.fill(tf.shape(val_anger), tf.mean(train_anger))

    mean_val = tf.fill(tf.shape(val_labels_valence), tf.reduce_mean(train_labels_valence))
    mean_ar = tf.fill(tf.shape(val_labels_arousal), tf.reduce_mean(train_labels_arousal))

    


    
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    
    
    
    conu_loss = bce(val_confusion, mean_confusion)
    h_loss = bce(val_happiness, mean_hapiness)
    ci_loss = bce(val_connfidence, mean_confidence)
    ca_loss = bce(val_calmness, mean_calmness)
    di_loss = bce(val_distress, mean_distress)
    f_loss = bce(val_fear, mean_fear)
    a_loss = bce(val_anger, mean_anger)
    val_loss = mse(val_labels_valence, mean_val)
    ar_loss = mse(val_labels_arousal, mean_ar)
    
    loss = conu_loss + h_loss + ci_loss + ca_loss + di_loss + f_loss + a_loss + val_loss * 10.0 + ar_loss * 10.0
    
    report = {
        "total_loss": loss.numpy().item(),
        "confusion_bce": conu_loss.numpy().item(),
        "happiness_bce": h_loss.numpy().item(),
        "confidence_bce": ci_loss.numpy().item(),
        "calmness_bce": ca_loss.numpy().item(),
        "distress_bce": di_loss.numpy().item(),
        "fear_bce": f_loss.numpy().item(),
        "anger_bce": a_loss.numpy().item(),
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