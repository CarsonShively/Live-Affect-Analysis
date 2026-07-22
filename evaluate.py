from pathlib import Path
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download, get_token, HfApi
from live_affect_analysis.gated_feature_fusion import GatedFeatureFusion
import json

def evaluate():
    eval_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/eval_matrices")
    
    test_images = np.load(eval_path / "test_images.npy")
    test_labels = np.load(eval_path / "test_labels.npy")
    
    
    test_images_tensor = tf.convert_to_tensor(test_images, dtype=tf.uint8)
    
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
        "category": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "satisfaction": tf.keras.losses.MeanSquaredError(),
        "calmness": tf.keras.losses.MeanSquaredError(),
        "valence": tf.keras.losses.MeanSquaredError(),
        "arousal": tf.keras.losses.MeanSquaredError(),
        "dominance": tf.keras.losses.MeanSquaredError()
    }
    
    loss_weights = {
        "category": 1.0,
        "satisfaction": 5.0,
        "calmness": 5.0,
        "valence": 5.0,
        "arousal": 5.0,
        "dominance": 5.0
    }
    
    model.compile(
        loss=losses,
        loss_weights=loss_weights
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
            path_in_repo="benchmark.json"
        )
if __name__ == "__main__":
    evaluate()