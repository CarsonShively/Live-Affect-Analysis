from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download, get_token, HfApi
from live_affect_analysis.gated_feature_fusion import GatedFeatureFusion
import json

def evaluate():
    eval_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/eval_matrices")
    
    test_images = np.load(eval_path / "test_images.npy")
    test_labels = np.load(eval_path / "test_labels.npy")
    
    test_len = test_labels.shape[0]
    
    test_images_tensor = tf.convert_to_tensor(test_images, dtype=tf.uint8)
    
    test_images_tensor = test_images_tensor[:test_len]
    
    test_labels_tensors = {
        "category": tf.convert_to_tensor(test_labels[:, :26], dtype=tf.float32),
        "valence": tf.convert_to_tensor(test_labels[:, 26:27], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(test_labels[:, 27:28], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(test_labels[:, 28:29], dtype=tf.float32),
        "gender": tf.convert_to_tensor(test_labels[:, 29:30], dtype=tf.float32),
        "age": tf.convert_to_tensor(test_labels[:, 30], dtype=tf.int32)
    }
    
    model_weights_path = Path(snapshot_download(
        repo_id="Carson-Shively/Affect-Analysis",
        repo_type="model",
        allow_patterns="gated_feature_fusion.weights.h5"
    ))
    
    model = GatedFeatureFusion()
    
    dummy_input = tf.zeros(shape=(1, 224, 224, 3))
    
    model(dummy_input)
    
    model.load_weights(model_weights_path / "gated_feature_fusion.weights.h5")
    
    losses = {
        "category": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError(),
        "dominance": tf.keras.losses.MeanSquaredError(),
        "gender": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "age": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    
    
    model.compile(
        loss=losses,
        jit_compile=False
    )
    
    out = model.evaluate(
        test_images_tensor,
        test_labels_tensors,
        batch_size=32,
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