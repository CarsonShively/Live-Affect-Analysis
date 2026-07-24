from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download, get_token, HfApi
from live_affect_analysis.low_latency_model import LowLatencyModel
import json

def evaluate():
    eval_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/eval_matrices")
    
    test_images = np.load(eval_path / "test_images.npy")
    test_labels = np.load(eval_path / "test_labels.npy")
    
    test_len = test_labels.shape[0]
    
    test_labels_valence = (test_labels[:, 26:27] - 1.0) / 9.0
    test_labels_arousal = (test_labels[:, 27:28] - 1.0) / 9.0
    
    test_happiness = test_labels[:, 5:6]
    test_calmness = test_labels[:, 9:10]
    test_sadness = test_labels[:, 12:13]
    test_fear = test_labels[:, 16:17]
    test_anger = test_labels[:, 22:23]
    
    test_images_tensor = tf.convert_to_tensor(test_images, dtype=tf.uint8)
    
    test_images_tensor = test_images_tensor[:test_len]
    
    test_labels_tensors = {
        "happiness": tf.convert_to_tensor(test_happiness, dtype=tf.float32),
        "calmness": tf.convert_to_tensor(test_calmness, dtype=tf.float32),
        "sadness": tf.convert_to_tensor(test_sadness, dtype=tf.float32),
        "fear": tf.convert_to_tensor(test_fear, dtype=tf.float32),
        "anger": tf.convert_to_tensor(test_anger, dtype=tf.float32),
        "valence": tf.convert_to_tensor(test_labels_valence, dtype=tf.float32),
        "arousal": tf.convert_to_tensor(test_labels_arousal, dtype=tf.float32)
    }
    
    model_weights_path = Path(snapshot_download(
        repo_id="Carson-Shively/Affect-Analysis",
        repo_type="model",
        allow_patterns="low_latency_model.weights.h5"
    ))
    
    model = LowLatencyModel()
    
    dummy_input = tf.zeros(shape=(1, 224, 224, 3))
    
    model(dummy_input)
    
    model.load_weights(model_weights_path / "low_latency_model.weights.h5")
    
    losses = {
        "happiness": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "calmness": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "sadness": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "fear": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "anger": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError()
    }
    
    loss_weights = {
        "happiness": 1.0,
        "calmness": 1.0,
        "sadness": 1.0,
        "fear": 1.0,
        "anger": 1.0,
        "valence": 10.0,
        "arousal": 10.0
    }
    
    
    model.compile(
        loss=losses,
        loss_weights=loss_weights,
        jit_compile=False
    )
    
    out = model.evaluate(
        test_images_tensor,
        test_labels_tensors,
        batch_size=16,
        return_dict=True
    )
    

    report = {}
    
    for key, val in out.items():
        report[key] = float(val)
    
    out_path = Path("/content/evaluation_report.json")
    
    with open(out_path, "w") as con:
        json.dump(report, con)
        
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            path_or_fileobj=out_path,
            path_in_repo="evaluation_report.json"
        )
if __name__ == "__main__":
    evaluate()