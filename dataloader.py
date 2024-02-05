import os
import torch
import glob
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image

class MyDataset_lr(Dataset):
    def __init__(self, metadata,metadata2,root_dir, transform = None):
        self.metadata = pd.read_csv(metadata) #anchor_image file (list of satellite images)
        self.metadata2 = np.loadtxt(metadata2).reshape(1,-1) #corresponding number(ID) of the POI/geo-most adjacent satellite image in the anchor_image file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path = self.metadata.iloc[idx, 0]
        im=Image.open(self.root_dir+path+'.png')
        cp_idx=self.metadata2[0,idx]
        path_cp = self.metadata.iloc[int(cp_idx), 0]
        im_cp=Image.open(self.root_dir+path_cp+'.png')
        if self.transform:
            sample = self.transform(im)
            sample_cp=self.transform(im_cp)
            
        return sample,sample_cp  # the anchor_image and corresponding POI/geo-most adjacent satellite image
