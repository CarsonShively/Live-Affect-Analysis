import threading
import av
from ultralytics import YOLO
import tensorflow as tf
from live_affect_analysis.low_latency_model import LowLatencyModel
import cv2
import json
from huggingface_hub import snapshot_download
from pathlib import Path

class ProcessFrame():
    def __init__(self):
        self.detector = YOLO("yolo26n.pt")
        
        model_folder = Path(snapshot_download(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            allow_patterns=["category_dictionary.json", "low_latency_model.weights.h5"]
        ))
        
        with open(model_folder / "category_dictionary.json", "r") as con:
            dictionary = json.load(con)
        self.dictionary_list = list(dictionary.keys())
        
        self.model = LowLatencyModel()
        dummy_input = tf.zeros(shape=[1, 224, 224, 3])
        self.model(x=dummy_input)
        self.model.load_weights(model_folder / "low_latency_model.weights.h5")

    
    def detect_person(self, image):
        result = self.detector(source=image, classes=[0], conf=0.4, verbose=False, imgsz=320)
        result = result[0]
        largest_box = 0.0
        largest_coords = None
        
        if result.boxes == 0:
            return None
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            size = (x2 - x1) * (y2 - y1)
            if size > largest_box:
                largest_coords = (int(x1), int(y1), int(x2), int(y2))
                largest_box = size
            
        return largest_coords
    
    def inference(self, image):
        try:
            
            cords = self.detect_person(image)
            if cords == None:
                output = {
                    "status": "No Person Detected"
                }
                
                
                return output
                
            else:
                x1, y1, x2, y2 = cords
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image[y1:y2, x1:x2]
                image = tf.convert_to_tensor(image, dtype=tf.uint8)
                image = tf.image.resize_with_pad(
                        image,
                        target_height=224,
                        target_width=224
                    )
                image = image[tf.newaxis, ...]
                output_dict = self.model(image)
                happiness = tf.sigmoid(output_dict["happiness"])
                calmness = tf.sigmoid(output_dict["calmness"])
                sadness = tf.sigmoid(output_dict["sadness"])
                fear = tf.sigmoid(output_dict["fear"])
                anger = tf.sigmoid(output_dict["anger"])
                valence = output_dict["valence"] * 9 + 1
                arousal = output_dict["arousal"] * 9 + 1
                
                output = {
                    "success": True,
                    "status": "Person Detected",
                    "mood": int(round(valence.numpy().item())),
                    "energy": int(round(arousal.numpy().item())),
                    "happiness": round(happiness.numpy().item() * 100, 2),
                    "calmness": round(calmness.numpy().item() * 100, 2),
                    "sadness": round(sadness.numpy().item() * 100, 2),
                    "fear": round(fear.numpy().item() * 100, 2),
                    "anger": round(anger.numpy().item() * 100, 2),
                }
                
                
                return output
            
        except Exception as error:
            print(f"Inference failed: {error}")
                    
                    