from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

class MiniPlacesDataset(Dataset): 
    def __init__(self, **kwargs):
        # load in the data 
        self.photos_path = kwargs['photos_path']
        self.labels_path = kwargs['labels_path']
        self.transform = kwargs['transform']
        self.load_size = 224
        self.images = []
        self.labels = []

        # read the text file
        with open(self.labels_path, 'r') as f: 
            for line in f:
                path, label = line.strip().split(" ")
                self.images.append(path)
                self.labels.append(label)

        self.images = np.array(self.images, np.object)
        self.labels = np.array(self.labels, np.int64)
        print("# images found at path '%s': %d" % (self.labels_path, self.images.shape[0]))

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx): 
        image = Image.open(os.path.join(self.photos_path, self.images[idx]))
        image = self.transform(image)
        # label is the index of the correct category
        label = self.labels[idx]
        return (image, label)
