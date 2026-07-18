from pathlib import Path
import tensorflow as tf

def benchmark():
    train_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    train_labels = train_path / "train_labels.py"
    val_labels = train_path / "val_labels.py"
    
    train = {
        "category": tf.convert_to_tensor(train_labels[:, 0], dtype=tf.int32),
        "satisfaction": tf.convert_to_tensor(train_labels[:, 1:2], dtype=tf.float32),
        "calmness": tf.convert_to_tensor(train_labels[:, 2:3], dtype=tf.float32),
        "valence": tf.convert_to_tensor(train_labels[:, 3:4], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(train_labels[:, 4:5], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(train_labels[:, 5:6], dtype=tf.float32)
    }
    
    mean = {
        "category": tf.convert_to_tensor(train_labels[:, 0], dtype=tf.int32),
        "satisfaction": tf.convert_to_tensor(train_labels[:, 1:2], dtype=tf.float32),
        "calmness": tf.convert_to_tensor(train_labels[:, 2:3], dtype=tf.float32),
        "valence": tf.convert_to_tensor(train_labels[:, 3:4], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(train_labels[:, 4:5], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(train_labels[:, 5:6], dtype=tf.float32)
    }