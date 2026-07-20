from pathlib import Path
from scipy.io import loadmat
from huggingface_hub import get_token, HfApi
import json
import numpy as np

def build_categories_dictionary():
    emotica_path = Path("/content/drive/MyDrive/Live-Affect-Analysis/emotica")
    
    annotations_path = emotica_path / "annotations/Annotations.mat"
    
    annotations = loadmat(annotations_path, squeeze_me=True, struct_as_record=False)
    
    train = annotations["train"]
    
    dictionary = {}
    dictionary_id = 0
    
    
    for sample in np.atleast_1d(train):
        for person in np.atleast_1d(sample.person):
            for category in np.atleast_1d(person.annotations_categories.categories):
                category = str(category)
                if not category in dictionary:
                    dictionary[category] = dictionary_id
                    dictionary_id += 1
    
    print(f"dictionary length: {len(dictionary)}")
    
    out_path = Path("/content/category_dictionary.json")
    
    with open(out_path, "w") as con:
        json.dump(dictionary, con)
    
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/Affect-Analysis",
            repo_type="model",
            path_or_fileobj=out_path,
            path_in_repo="category_dictionary.json"
        )
    
if __name__ == "__main__":
    build_categories_dictionary()