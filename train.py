import tensorflow as tf
from pathlib import Path
import numpy as np
from live_affect_analysis.gated_feature_fusion import GatedFeatureFusion
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
    
    train_labels = np.load(train_path / "train_labels.npy")
    val_labels = np.load(train_path / "val_labels.npy")
    
    train_labels_tensors = {
        "category": tf.convert_to_tensor(train_labels[:, 0], dtype=tf.int32),
        "satisfaction": tf.convert_to_tensor(train_labels[:, 1:2], dtype=tf.float32),
        "calmness": tf.convert_to_tensor(train_labels[:, 2:3], dtype=tf.float32),
        "valence": tf.convert_to_tensor(train_labels[:, 3:4], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(train_labels[:, 4:5], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(train_labels[:, 5:6], dtype=tf.float32)
    }
    
    val_labels_tensors = {
        "category": tf.convert_to_tensor(val_labels[:, 0], dtype=tf.int32),
        "satisfaction": tf.convert_to_tensor(val_labels[:, 1:2], dtype=tf.float32),
        "calmness": tf.convert_to_tensor(val_labels[:, 2:3], dtype=tf.float32),
        "valence": tf.convert_to_tensor(val_labels[:, 3:4], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(val_labels[:, 4:5], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(val_labels[:, 5:6], dtype=tf.float32)
    }
    
    del train_labels, val_labels
    
    print(f"train images shape: {train_images_tensor.shape}")
    print(f"val images shape: {val_images_tensor.shape}")
    
    model = GatedFeatureFusion()
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    losses = {
        "category": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "satisfaction": tf.keras.losses.MeanSquaredError(),
        "calmness": tf.keras.losses.MeanSquaredError(),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError(),
        "dominance": tf.keras.losses.MeanSquaredError()
    }
    
    loss_weights = {
        "category": 1,
        "satisfaction": 1,
        "calmness": 1,
        "valence": 1,
        "arousal": 1,
        "dominance": 1
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
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
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping]
    )
    
    model_out = Path("/content/model")
    if model_out.is_dir():
        shutil.rmtree(model_out)
    model_out.mkdir(parents=True, exist_ok=True)    
    
    model.save_weights(model_out / "gated_feature_fusion.weights.h5")
    
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            path_or_fileobj=model_out / "gated_feature_fusion.weights.h5",
            path_in_repo="gated_feature_fusion.weights.h5"
        )
    
if __name__ == "__main__":
    train_model()