from scipy.io import loadmat, whosmat
from pathlib import Path

def inspect_annotations():
    mat_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica/annotations/Annotations.mat")
    
    print(whosmat(mat_path))
    
    annotations = loadmat(mat_path, structs_as_record=False, squeeze_me=True)
     
    train = annotations["train"]
    
    print(f"train type: {type(train)}")
    print(f"train shape: {train.shape}")
     
if __name__ == "__main__":
    inspect_annotations()