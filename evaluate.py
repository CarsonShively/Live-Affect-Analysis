from pathlib import Path
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download, get_token, HfApi
from live_affect_analysis.affect_analysis_cnn import AffectAnalysisCNN
import json

def evaluate():
    eval_path = Path("/content/drive/MyDrive/Affect-Analysis/evel_matrices")
    
    test_images = np.load(eval_path / "test_images.npy")
    test_labels = np.load(eval_path / "test_labels.npy")
    
    test_images_tensor = tf.convert_to_tensor(test_images, dtype=tf.uint8)
    
    test_labels_tensors = {
        "category": tf.convert_to_tensor(test_labels[:, 0], dtype=tf.int32),
        "satisfaction": tf.convert_to_tensor(test_labels[:, 1:2], dtype=tf.float32),
        "calmness": tf.convert_to_tensor(test_labels[:, 2:3], dtype=tf.float32),
        "valence": tf.convert_to_tensor(test_labels[:, 3:4], dtype=tf.float32),
        "arousal": tf.convert_to_tensor(test_labels[:, 4:5], dtype=tf.float32),
        "dominance": tf.convert_to_tensor(test_labels[:, 5:6], dtype=tf.float32)
    }
    
    model_weights_path = Path(snapshot_download(
        repo_id="Carson-Shively/Affect-Analysis",
        repo_type="model",
        allow_patterns="affect_analysis.weights.h5"
    ))
    
    model = AffectAnalysisCNN()
    
    dummy_input = tf.zeros(shape=(1, 224, 224, 3))
    
    model(dummy_input)
    
    model.load_weights(model_weights_path / "affect_analysis.weights.h5")
    
    out = model(test_images_tensor)
    
    loss = model.compute_loss(
        x=None,
        y_pred=out,
        y=test_labels_tensors
    )
    
    report = {"holdout_loss": loss.numpy().item()}
    
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