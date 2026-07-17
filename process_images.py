from pathlib import Path
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

def process_images():
    images_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/Data")
    labels_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/Labels/HECO_Labels.csv")
    
    local_images_path = Path("/content/Data")
    
    if not local_images_path.is_dir():
        shutil.copytree(
            images_path,
            local_images_path,
            dirs_exist_ok=True
        )
    
    df = pd.read_csv(labels_path)
    
    print("data mounted")
    
    images_count = 0
    
    for image in images_path.iterdir():
        if image.is_file():
            images_count += 1
            
    print(f"images count: {images_count}")
    
    out_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/processed_images")
    if out_path.is_dir():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    train_index = int(images_count * 0.70)
    val_index = int(images_count * 0.80)

    train = []
    val = []
    test = []
    labels = []

    counter = 0
    for image_path in local_images_path.iterdir():
        try:
            if not image_path.is_file():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            label_row = df[df["Image"] == image_path.name]
            x_min = int(label_row["xmin"].iloc[0])
            x_max = int(label_row["xmax"].iloc[0])
            y_min = int(label_row["ymin"].iloc[0])
            y_max = int(label_row["ymax"].iloc[0])
            cropped_image = image[y_min:y_max, x_min:x_max]
            image_tensor = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
            image_tensor_resized = tf.image.resize_with_pad(
                image_tensor,
                target_height=224,
                target_width=224
            )
            
            image_tensor_normalized = image_tensor_resized / 255.0
            
            image_numpy = image_tensor_normalized.numpy()
            
            if counter < train_index:
                train.append(image_numpy)
            elif counter < val_index:
                val.append(image_numpy)
            else:
                test.append(image_numpy)
            counter += 1
            images_count -= 1
            print(f"{images_count} images remaining")
        except Exception as error:
            print(f"Error file: {image_path.name}: {error}")
            continue
        
    train_numpy = np.stack(train)
    del train
    val_numpy = np.stack(val)
    del val
    test_numpy = np.stack(test)
    del test 
    
    local_out_path = Path(__file__).resolve().parents[0] / "processed_images"
    if local_images_path.is_dir():
        shutil.rmtree(local_out_path)
    local_out_path.mkdir(parents=True, exist_ok=True)
        
    np.save(local_out_path / "train.npy", train_numpy)  
    np.save(local_out_path / "val.npy", val_numpy)  
    np.save(local_out_path / "test.npy", test_numpy) 
    
    np.save(out_path / "train.npy", train_numpy)  
    np.save(out_path / "val.npy", val_numpy)  
    np.save(out_path / "test.npy", test_numpy)  
        
if __name__ == "__main__":
    process_images()