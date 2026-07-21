import threading
import av
from ultralytics import YOLO
import tensorflow as tf
from live_affect_analysis.gated_feature_fusion import GatedFeatureFusion
import cv2
import json
from huggingface_hub import snapshot_download
from pathlib import Path

class ProcessFrame():
    def __init__(self):
        self.lock = threading.Lock()
        self.inferencing = False
        self.prediction = None
        self.detector = YOLO("yolo26n.pt")
        
        model_folder = Path(snapshot_download(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            allow_patterns=["category_dictionary.json", "gated_feature_fusion.weights.h5"]
        ))
        
        with open(model_folder / "category_dictionary.json", "r") as con:
            dictionary = json.load(con)
        self.dictionary_list = list(dictionary.keys())
        
        self.model = GatedFeatureFusion()
        dummy_input = tf.zeros(shape=[1, 224, 224, 3])
        self.model(x=dummy_input)
        self.model.load_weights(model_folder / "gated_feature_fusion.weights.h5")
        self.status = None
    
    def detect_person(self, image):
        result = self.detector(source=image, classes=[0], conf=0.4, verbse=False, imgsz=320)
        result = result[0]
        largest_box = 0.0
        largest_coords = None
        
        if len(result.boxes == 0):
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
                self.status = "No Person Detected"
            else:
                self.status = "Person Detected"
                x1, y1, x2, y2 = cords
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image[y1:y2, x1:x2]
                image = tf.convert_to_tensor(image, dtype=tf.uint8)
                image = tf.image.resize_with_pad(
                        image,
                        target_height=224,
                        target_width=224
                    )
                output_dict = self.model(image)
                category_probabilites = tf.nn.sigmoid(output_dict["category"])
                
                top_values, top_indices = tf.math.top_k(category_probabilites, k=3)
                
                if output_dict["gender"] >= 0.5:
                    gender = "Male"
                else:
                    gender = "Female"
                
                age_index = tf.argmax(output_dict["age"], axis=-1)
                
                if age_index == 0:
                    age = "Kid"
                elif age_index == 1:
                    age = "Teenager"
                else:
                    age = "Adult"
                
                output = {
                    "age": age,
                    "gender": gender,
                    "valence": f"Mood                   {int(round(output_dict["valence"]))}/10",
                    "arousal": f"Energy                 {int(round(output_dict["arousal"]))}/10",
                    "dominance": f"Assertiveness        {int(round(output_dict["dominance"]))}/10",
                    "category_1": f"{self.dictionary_list[top_indices[0]]}      {top_values[0]}%",
                    "category_2": f"{self.dictionary_list[top_indices[1]]}      {top_values[1]}%",
                    "category_3": f"{self.dictionary_list[top_indices[2]]}      {top_values[2]}%"
                }
                
                self.prediction = output
            
        except Exception as error:
            print(f"Inference failed: {error}")
        finally:
            with self.lock():
                self.inferencing = False
        
        
    def build_overlay(self, image):
        x = 0
        
        
    def recv(self, frame):
        image = frame.to_ndarray(frame="bgr24")
        
        with self.lock():
            if not self.inferencing:
                should_score = True
                self.inferencing = True
            else:
                should_score = False                
            
        if should_score:
            self.inference(image)
            
        image = self.build_overlay(image)
            
        return av.VideoFrame.from_ndarray(image, format="bgr24")