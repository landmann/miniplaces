from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

class MiniPlacesTestLoader(Dataset): 
    def __init__(self, **kwargs): 
        self.photos_path = kwargs['photos_path']
        self.transform = kwargs['transform']
        self.num_files = len(os.listdir(self.photos_path))

    def __len__(self): 
        num_files = len(os.listdir(self.photos_path))
        print("Testing on {} images...".format(num_files))
        assert num_files == 10000
        return 10000

    def __getitem__(self, idx): # idx 0 to 999
        img_num = idx + 1
        img_name =  str(img_num).zfill(8) + '.jpg'
        path = self.photos_path + img_name
        image = Image.open(path)
        return self.transform(image), 'test/' + img_name
