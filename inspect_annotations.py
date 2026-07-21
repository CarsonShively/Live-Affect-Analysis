from scipy.io import loadmat, whosmat
from pathlib import Path

def inspect_annotations():
    mat_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica/annotations/Annotations.mat")
    
    print(whosmat(mat_path))
    
    annotations = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
     
    test = annotations["test"]
    
    print(test[0].person.annotations_continuous._fieldnames)
    
if __name__ == "__main__":
    inspect_annotations()