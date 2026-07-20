from pathlib import Path
from scipy import loadmat
import cv2
import tensorflow as tf

def build_matrices():
    emotica_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica")
    annotations_path = emotica_path / "annotations/Annotations.mat"
    
    annotations = loadmat(annotations_path, squeeze_me=True, struct_as_record=False)
    
    train = annotations["train"]
    train_len = train.shape[0]
    
    for i in range(train_len):
        sample = train[i]
        image_folder = sample.folder
        image_file = sample.filename
        image_path = emotica_path / "images" / image_folder / image_file
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        left, top, right, bottom = sample.person.body_bbox
        
        image = image[top:bottom, left:right]
    
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor_resized = tf.image.resize_with_pad(
            image_tensor,
            target_height=224,
            target_width=224
        )
        image_numpy = tf.cast(
            tf.round(image_tensor_resized),
            tf.uint8
        ).numpy()
    
if __name__ == "__main__":
    build_matrices()