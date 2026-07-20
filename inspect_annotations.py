from scipy.io import loadmat, whosmat
from pathlib import Path

def inspect_annotations():
    mat_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica/annotations/Annotations.mat")
    
    print(whosmat(mat_path))
    
    annotations = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
     
    train = annotations["train"]
    
    print(f"train type: {type(train)}")
    print(f"train shape: {train.shape}")
    print(f"train item type {type(train[0])}")
    print(f"train item fields {train[0]._fieldnames}")
    
    print(f"filename {type(train[0].filename)}")
    print(f"folder {type(train[0].folder)}")
    print(f"image size {type(train[0].image_size)}")
    print(f"original database {type(train[0].original_database)}")
    print(f"person {type(train[0].person)}")
     
    print(f"image size fields {(train[0].image_size)._fieldnames}")
    print(f"original database fields {(train[0].original_database)._fieldnames}")
    print(f"person fields {(train[0].person)._fieldnames}")
    
    print(f"image size ncol nrow types {type((train[0].image_size).n_col)}, {type((train[0].image_size).n_row)}")
    print(f"original databse field types {type((train[0].original_database).name)}, {type((train[0].original_database).info)}")
    
    print(f"person field types {type((train[0].person).body_bbox)}, {type((train[0].person).annotations_categories)}, {type((train[0].person).annotations_continuous)}, {type((train[0].person).gender)}, {type((train[0].person).age)}")
    
    print(f"bbox item {type(((train[0].person).body_bbox)[0])}")
    print(f"bbox len {len((train[0].person).body_bbox)}")
    
    print(f"cat fields {((train[0].person).annotations_categories)._fieldnames}")
    print(f"cat type {type(((train[0].person).annotations_categories).categories)}")
    
if __name__ == "__main__":
    inspect_annotations()