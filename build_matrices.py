from pathlib import Path
from scipy.io import loadmat
import cv2
import tensorflow as tf
import shutil
from huggingface_hub import snapshot_download
import json
import numpy as np

def build_matrices():
    
    dictionary_path = Path(snapshot_download(
        repo_id="Carson-Shively/Affect-Analysis",
        repo_type="model",
        allow_patterns="category_dictionary.json"
    ))
    
    with open(dictionary_path / "category_dictionary.json", "r") as con:
        dictionary = json.load(con)
    
    dictionary_len = len(dictionary)
    
    label_len = dictionary_len + 5
    
    drive_emotica_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica")
    
    local_emotica_path = Path("/content/emotica")
    
    if not local_emotica_path.is_dir():
        shutil.copytree(
            drive_emotica_path,
            local_emotica_path
        )
    
    annotations_path = local_emotica_path / "annotations/Annotations.mat"
    
    annotations = loadmat(annotations_path, squeeze_me=True, struct_as_record=False)
    
    train = annotations["train"]
    val = annotations["val"]
    test = annotations["test"]
    
    train_len = 0
    for sample in train:
        for person in np.atleast_1d(sample.person):
            train_len += 1
    
    val_len = 0
    for sample in val:
        for person in np.atleast_1d(sample.person):
            val_len += 1
            
    test_len = 0
    for sample in test:
        for person in np.atleast_1d(sample.person):
            test_len += 1
    
    local_out = Path("/content/matrices")
    
    if local_out.is_dir():
        shutil.rmtree(local_out)
        
    local_out.mkdir(parents=True, exist_ok=True)
    
    train_images = np.lib.format.open_memmap(
        local_out / "train.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(train_len, 224, 224, 3)
    )
    
    val_images = np.lib.format.open_memmap(
        local_out / "val.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(val_len, 224, 224, 3)
    )
    test_images = np.lib.format.open_memmap(
        local_out / "test.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(test_len, 224, 224, 3)
    )
    
    train_labels = []
    val_labels = []
    test_labels = []
    
    splits = [(train, train_images, train_labels, "train"), (val, val_images, val_labels, "val"), (test, test_images, test_labels, "test")]
    for split, image_out, label_out, split_string in splits:
        total = split.shape[0]
        print_remaining = 500
        append_index = 0
        split_skip_count = 0
        for index, sample in enumerate(split):
            image_folder = sample.folder
            image_file = sample.filename
            image_path = local_emotica_path / "images" / image_folder / image_file
            
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

            for person in np.atleast_1d(sample.person):
            
                left, top, right, bottom = person.body_bbox
                
                left = int(left)
                top = int(top)
                right = int(right)
                bottom = int(bottom)
                
                if left < 0 or right > width or top < 0 or bottom > height or right <= left or bottom <= top:
                    split_skip_count += 1
                    print(f"{split_string} skip count: {split_skip_count}")
                    continue
                
                image_cropped = image[top:bottom, left:right]
            
                image_tensor = tf.convert_to_tensor(image_cropped, dtype=tf.float32)
                image_tensor_resized = tf.image.resize_with_pad(
                    image_tensor,
                    target_height=224,
                    target_width=224
                )
                image_numpy = tf.cast(
                    tf.round(image_tensor_resized),
                    tf.uint8
                ).numpy()
                
                labels = np.zeros(shape=label_len, dtype=np.float32)

                
                if split_string == "train":
                    for category in np.atleast_1d(person.annotations_categories.categories):
                        category = str(category)
                        if category in dictionary:
                            labels[dictionary[category]] = 1
                        else:
                            continue
                else:
                    for category in np.atleast_1d(person.combined_categories):
                        category = str(category)
                        if category in dictionary:
                            labels[dictionary[category]] = 1
                        else:
                            continue
                
                if split_string == "train":
                    labels[dictionary_len] = person.annotations_continuous.valence
                    labels[dictionary_len+1] = person.annotations_continuous.arousal
                    labels[dictionary_len+2] = person.annotations_continuous.dominance
                else:
                    labels[dictionary_len] = person.combined_continuous.valence
                    labels[dictionary_len+1] = person.combined_continuous.arousal
                    labels[dictionary_len+2] = person.combined_continuous.dominance
                
                if person.gender == "Male":
                    labels[dictionary_len+3] = 1
                elif person.gender == "Female":
                    labels[dictionary_len+3] = 0
                else:
                    raise ValueError(f"gender value error sample: {index}")
                
                if person.age == "Kid":
                    labels[dictionary_len+4] = 0
                elif person.age == "Teenager":
                    labels[dictionary_len+4] = 1
                elif person.age == "Adult":
                    labels[dictionary_len+4] = 2
                else:
                    raise ValueError(f"age value error sample: {index}")
                
                image_out[append_index] = image_numpy
                append_index += 1
                label_out.append(labels)
                
                
            if index == print_remaining:
                print(f"{split_string} remaining: {total-index-1}")
                print_remaining += 500
        
        
    train_images.flush()
    val_images.flush()
    test_images.flush()
        
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
        local_out / "test.npy",
        drive_eval_out / "test_images.npy"
    )
    
    print("test images complete")
    
    np.save(drive_eval_out / "test_labels.npy", test_labels_matrix)
    print("test labels complete")
    print("complete")
    
if __name__ == "__main__":
    build_matrices()