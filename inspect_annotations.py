from scipy.io import whosmat
from pathlib import Path

def inspect_annotations():
    mat_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica/annotations/Annotations.mat")
    
    print(whosmat(mat_path))
     
if __name__ == "__main__":
    inspect_annotations()