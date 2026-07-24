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
    
    train_labels_tensors = {
        "confusion": tf.convert_to_tensor(train_confusion, dtype=tf.float32),
        "happiness": tf.convert_to_tensor(train_happiness, dtype=tf.float32),
        "confidence": tf.convert_to_tensor(train_connfidence, dtype=tf.float32),
        "calmness": tf.convert_to_tensor(train_calmness, dtype=tf.float32),
        "distress": tf.convert_to_tensor(train_distress, dtype=tf.float32),
        "fear": tf.convert_to_tensor(train_fear, dtype=tf.float32),
        "anger": tf.convert_to_tensor(train_anger, dtype=tf.float32),
        "valence": tf.convert_to_tensor(train_labels_valence, dtype=tf.float32),
        "arousal": tf.convert_to_tensor(train_labels_arousal, dtype=tf.float32)
    }
    
    val_labels_tensors = {
        "confusion": tf.convert_to_tensor(val_confusion, dtype=tf.float32),
        "happiness": tf.convert_to_tensor(val_happiness, dtype=tf.float32),
        "confidence": tf.convert_to_tensor(val_connfidence, dtype=tf.float32),
        "calmness": tf.convert_to_tensor(val_calmness, dtype=tf.float32),
        "distress": tf.convert_to_tensor(val_distress, dtype=tf.float32),
        "fear": tf.convert_to_tensor(val_fear, dtype=tf.float32),
        "anger": tf.convert_to_tensor(val_anger, dtype=tf.float32),
        "valence": tf.convert_to_tensor(val_labels_valence, dtype=tf.float32),
        "arousal": tf.convert_to_tensor(val_labels_arousal, dtype=tf.float32)
    }
    
    del train_labels, val_labels
    
    print(f"train images shape: {train_images_tensor.shape}")
    print(f"val images shape: {val_images_tensor.shape}")
    
    model = LowLatencyModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    losses = {
        "confusion": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "happiness": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "confidence": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "calmness": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "distress": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "fear": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "anger": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError()
    }
    
    loss_weights = {
        "confusion": 1.0,
        "happiness": 1.0,
        "confidence": 1.0,
        "calmness": 1.0,
        "distress": 1.0,
        "fear": 1.0,
        "anger": 1.0,
        "valence": 10.0,
        "arousal": 10.0
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        jit_compile=False,
        loss_weights=loss_weights
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