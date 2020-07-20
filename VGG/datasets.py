### torch lib
import torch
from torch.utils.data import Dataset

from glob import glob
import numpy as np
from PIL import Image

# Download the data from https://download.pytorch.org/tutorial/hymenoptera_data.zip

class HymenopteraDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = glob(root+'/*/*')
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        image = np.asarray(Image.open(self.files[idx]).convert('RGB'))
        if self.transform is not None:
            image = self.transform(image)
        label = self.files[idx].split('/')[-2]
        return image, label
