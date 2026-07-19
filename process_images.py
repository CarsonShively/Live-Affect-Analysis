from pathlib import Path
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
from live_affect_analysis.map import category_map, satisfaction_map, calmness_map


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
    
    samples_count = len(df)
    
    train_size = int(samples_count * 0.70)
    val_size = int(samples_count * 0.10)
    test_size = samples_count - train_size - val_size 
    
    local_out = Path("/content/matrices")
    if local_out.is_dir():
        shutil.rmtree(local_out)
    local_out.mkdir(parents=True, exist_ok=True)
    
    train_images = np.lib.format.open_memmap(
        local_out / "train.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(train_size, 224, 224, 3)
    )
    
    val_images = np.lib.format.open_memmap(
        local_out / "val.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(val_size, 224, 224, 3)
    )
    test_images = np.lib.format.open_memmap(
        local_out / "test.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(test_size, 224, 224, 3)
    )
    
    train_labels = []
    val_labels = []
    test_labels = []
       
    print(f"samples count: {samples_count}")

    train_counter = 0
    val_counter = 0
    test_counter = 0
    for _, row in df.iterrows():
        try:
            image_path = local_images_path / row["Image"]
            
            if not image_path.is_file():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            x_min = int(row["xmin"])
            x_max = int(row["xmax"])
            y_min = int(row["ymin"])
            y_max = int(row["ymax"])
            cropped_image = image[y_min:y_max, x_min:x_max]
            image_tensor = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
            image_tensor_resized = tf.image.resize_with_pad(
                image_tensor,
                target_height=224,
                target_width=224
            )
            
            image_numpy = tf.cast(
                tf.round(image_tensor_resized),
                tf.uint8
            ).numpy()
            
            labels = []

            labels.append(category_map[row["Category"]])
            labels.append(satisfaction_map[row["Label_SA"]])
            labels.append(calmness_map[row["Label_CA"]])
            labels.append(row["Valence"])
            labels.append(row["Arousal"])
            labels.append(row["Dominance"])

            if train_counter < train_size:
                train_images[train_counter] = image_numpy
                train_counter += 1
                print(f"train remaining: {train_size - train_counter}")
                train_labels.append(labels)
            elif val_counter < val_size:
                val_images[val_counter] = image_numpy
                val_counter += 1
                print(f"val remaining: {val_size - val_counter}")
                val_labels.append(labels)
            else:
                test_images[test_counter] = image_numpy
                test_counter += 1
                test_remaining = test_size - test_counter
                print(f"test remaining: {test_remaining}")
                test_labels.append(labels)

        except Exception as error:
            print(f"Error file: {image_path.name}: {error}")
            continue
        
    test_images.flush()
        
    trimmed_test = np.lib.format.open_memmap(
        local_out / "test_trimmed.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(test_counter, 224, 224, 3)
    )
    
    trimmed_test[:] = test_images[:test_counter]
    
    train_images.flush()
    val_images.flush()
    trimmed_test.flush()
        
    train_labels_matrix = np.array(train_labels)
    val_labels_matrix = np.array(val_labels)
    test_labels_matrix = np.array(test_labels)
    
    drive_modeling_out = Path("/content/drive/MyDrive/Live-Affect-Analysis/modeling_matrices")
    drive_eval_out = Path("/content/drive/MyDrive/Live-Affect-Analysis/eval_matrices")
    
    if drive_modeling_out.is_dir():
        shutil.rmtree(drive_modeling_out)
        
    if drive_eval_out.is_dir():
        shutil.rmtree(drive_eval_out)
        
    drive_modeling_out.mkdir(parents=True, exist_ok=True)
    drive_eval_out.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(
        local_out / "train.npy",
        drive_modeling_out / "train_images.npy"
    )
    
    print("train images complete")
    
    shutil.copy2(
        local_out / "val.npy",
        drive_modeling_out / "val_images.npy"
    )
    
    print("val images complete")
    
    np.save(drive_modeling_out / "train_labels.npy", train_labels_matrix)
    np.save(drive_modeling_out / "val_labels.npy", val_labels_matrix)
    print("train and val labels complete")
    
    shutil.copy2(
        local_out / "test_trimmed.npy",
        drive_eval_out / "test_images.npy"
    )
    
    print("test images complete")
    
    np.save(drive_eval_out / "test_labels.npy", test_labels_matrix)
    print("test labels complete")
    print("complete")
        
if __name__ == "__main__":
    process_images()