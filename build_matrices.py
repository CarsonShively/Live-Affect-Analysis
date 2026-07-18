from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from live_affect_analysis.map import category_map, satisfaction_map, calmness_map

def build_matrices():
    images_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/processed_images/")
    local_images_path = Path(__file__).resolve().parents[0] / "images"
    labels = Path("/content/drive/MyDrive/Live-Affect-Analysis/Labels/HECO_Labels.csv")
    
    df = pd.read_csv(labels)
    
    print("loaded")
    
    if not local_images_path.is_dir():
        shutil.copytree(
            images_path,
            local_images_path,
            dirs_exist_ok=True
        )
        
    count = 0
    for file in local_images_path.iterdir():
        if file.is_file():
            count += 1
            
    train_size = int(count * 0.70)
    val_size = int(count * 0.10)
    test_size = count - val_size - train_size
    
    local_out = Path("/content/image_matrices")
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
    
    total_samples = train_size + val_size + test_size
    samples_remaining = total_samples
    
    samples_counter = 0
    train_counter = 0
    val_counter = 0
    test_counter = 0
    
    for file in local_images_path.iterdir():
        name = file.stem
        image = np.load(file)
        
        labels = []
        
        label_df = df[df["Image"] == name]
        labels.append(category_map[label_df["Category"].iloc[0]])
        labels.append(satisfaction_map[label_df["Label_SA"].iloc[0]])
        labels.append(calmness_map[label_df["Label_CA"].iloc[0]])
        labels.append(label_df["Valence"].iloc[0])
        labels.append(label_df["Arousal"].iloc[0])
        labels.append(label_df["Dominance"].iloc[0])
        
        if samples_counter < train_size:
            train_images[train_counter] = image
            train_labels.append(labels)
            train_counter += 1
        elif samples_counter < train_size + val_size:
            val_images[val_counter] = image
            val_labels.append(labels)
            val_counter += 1
        elif samples_counter < total_samples:
            test_images[test_counter] = image
            test_labels.append(labels)
            test_counter += 1
        else:
            print("samples != samples spots")
            continue
            
            
        samples_counter += 1
        samples_remaining -= 1
        print(f"samples remaining: {samples_remaining}")
        
    train_images.flush()
    val_images.flush()
    test_images.flush()
        
    train_labels_matrix = np.stack(train_labels)
    val_labels_matrix = np.stack(val_labels)
    test_labels_matrix = np.stack(test_labels)
    
    print("copying to drive")
    
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
    
    shutil.copy2(
        local_out / "val.npy",
        drive_modeling_out / "val_images.npy"
    )
    
    np.save(drive_modeling_out / "train_labels.npy", train_labels_matrix)
    np.save(drive_modeling_out / "val_labels.npy", val_labels_matrix)
    
    shutil.copy2(
        local_out / "test.npy",
        drive_eval_out / "test_images.npy"
    )
    
    np.save(drive_eval_out / "test_labels.npy", test_labels_matrix)
    
    print("complete")
    
if __name__ == "__main__":
    build_matrices()