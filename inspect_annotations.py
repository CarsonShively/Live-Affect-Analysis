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
     
    print(f"image size {(train[0].image_size)._fieldnames}")
    print(f"original database {(train[0].original_database)._fieldnames}")
    print(f"person {(train[0].person)._fieldnames}")
if __name__ == "__main__":
    inspect_annotations()