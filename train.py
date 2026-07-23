import os
os.environ["MPLBACKEND"] = "Agg"
import tensorflow as tf
from pathlib import Path
import numpy as np
from live_affect_analysis.low_latency_model import LowLatencyModel
from huggingface_hub import get_token, HfApi
import shutil


def train_model():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    
    train_images = np.load(train_path / "train_images.npy")
    train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.uint8)
    del train_images
    
    val_images = np.load(train_path / "val_images.npy")
    val_images_tensor = tf.convert_to_tensor(val_images, dtype=tf.uint8)
    del val_images
    
    train_labels = np.load(train_path / "train_labels.npy").astype(np.float32)
    val_labels = np.load(train_path / "val_labels.npy").astype(np.float32)
    
    train_len = train_labels.shape[0]
    val_len = val_labels.shape[0]
    
    train_images_tensor = train_images_tensor[:train_len]
    val_images_tensor = val_images_tensor[:val_len]
    
    print(f"train images shape: {train_images_tensor.shape}")
    print(f"train labels shape: {train_labels.shape}")
    
    print(f"val images shape: {val_images_tensor.shape}")
    print(f"val labels shape: {val_labels.shape}")
    
    
    
    train_labels_tensors = {
        "category": tf.convert_to_tensor(train_labels[:, :26], dtype=tf.float32),
        "valence": tf.convert_to_tensor(train_labels[:, 26:27], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(train_labels[:, 27:28], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(train_labels[:, 28:29], dtype=tf.float32),
        "gender": tf.convert_to_tensor(train_labels[:, 29:30], dtype=tf.float32),
        "age": tf.convert_to_tensor(train_labels[:, 30], dtype=tf.int32)
    }
    
    val_labels_tensors = {
        "category": tf.convert_to_tensor(val_labels[:, :26], dtype=tf.float32),
        "valence": tf.convert_to_tensor(val_labels[:, 26:27], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(val_labels[:, 27:28], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(val_labels[:, 28:29], dtype=tf.float32),
        "gender": tf.convert_to_tensor(val_labels[:, 29:30], dtype=tf.float32),
        "age": tf.convert_to_tensor(val_labels[:, 30], dtype=tf.int32)
    }
    
    del train_labels, val_labels
    
    print(f"train images shape: {train_images_tensor.shape}")
    print(f"val images shape: {val_images_tensor.shape}")
    
    model = LowLatencyModel()
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    losses = {
        "category": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError(),
        "dominance": tf.keras.losses.MeanSquaredError(),
        "gender": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "age": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        jit_compile=False
    )
     
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        train_images_tensor,
        train_labels_tensors,
        validation_data=(val_images_tensor, val_labels_tensors),
        batch_size=16,
        epochs=50,
        callbacks=[early_stopping]
    )
    
    model_out = Path("/content/model")
    if model_out.is_dir():
        shutil.rmtree(model_out)
    model_out.mkdir(parents=True, exist_ok=True)    
    
    model.save_weights(model_out / "low_latency_model.weights.h5")
    
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            path_or_fileobj=model_out / "low_latency_model.weights.h5",
            path_in_repo="low_latency_model.weights.h5"
        )
    
if __name__ == "__main__":
    train_model()