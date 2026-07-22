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
    train_valence = tf.convert_to_tensor(train_labels[:, 26:27], dtype=tf.float32)
    train_arousal = tf.convert_to_tensor(train_labels[:, 27:28], dtype=tf.float32)
    train_dominance = tf.convert_to_tensor(train_labels[:, 28:29], dtype=tf.float32)
    train_gender = tf.convert_to_tensor(train_labels[:, 29:30], dtype=tf.float32)
    train_age = tf.convert_to_tensor(train_labels[:, 30], dtype=tf.int32)

    val_category = tf.convert_to_tensor(val_labels[:, :26], dtype=tf.float32)
    val_valence = tf.convert_to_tensor(val_labels[:, 26:27], dtype=tf.float32)
    val_arousal = tf.convert_to_tensor(val_labels[:, 27:28], dtype=tf.float32)
    val_dominance = tf.convert_to_tensor(val_labels[:, 28:29], dtype=tf.float32)
    val_gender = tf.convert_to_tensor(val_labels[:, 29:30], dtype=tf.float32)
    val_age = tf.convert_to_tensor(val_labels[:, 30], dtype=tf.int32)

    category_probabilities = tf.reduce_mean(train_category, axis=0)
    mean_categories = tf.broadcast_to(category_probabilities, val_category.shape)

    gender_prob = tf.reduce_mean(train_gender, axis=0)
    gender_mean = tf.broadcast_to(gender_prob, val_gender.shape)

    mean_val = tf.fill(tf.shape(val_valence), tf.reduce_mean(train_valence))
    mean_ar = tf.fill(tf.shape(val_arousal), tf.reduce_mean(train_arousal))
    mean_dom = tf.fill(tf.shape(val_dominance), tf.reduce_mean(train_dominance))
    

    age_classes = tf.math.bincount(
        train_age,
        minlength=3,
        maxlength=3,
        dtype=tf.float32
    )
    
    age_prob = age_classes / tf.reduce_sum(age_classes)
    mean_age = tf.broadcast_to(
        age_prob,
        (tf.shape(val_age)[0], 3)
    )
    
    
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    
    
    
    cat_loss = bce(val_category, mean_categories)
    val_loss = mse(val_valence, mean_val)
    ar_loss = mse(val_arousal, mean_ar)
    dom_loss = mse(val_dominance, mean_dom)
    gender_loss = bce(val_gender, gender_mean)
    age_loss = sce(val_age, mean_age)
    
    loss = cat_loss + gender_loss + age_loss + val_loss + ar_loss + dom_loss
    
    report = {
        "total_loss": loss.numpy().item(),
        "category_bce": cat_loss.numpy().item(),
        "valence_mse": val_loss.numpy().item(),
        "arousal_mse": ar_loss.numpy().item(),
        "dominance": dom_loss.numpy().item(),
        "gender_bce": gender_loss.numpy().item(),
        "age_sce": age_loss.numpy().item()
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