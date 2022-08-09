from random import sample
import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
# SimCLR
import math
from sklearn.decomposition import PCA
from PIL import Image


POI=np.loadtxt('POI_array_cate_1st_duiqi.txt') # POI array, following the order of satellite image list
max_idx_list=[] 
cos_list=[]
for test_idx in range(POI.shape[0]):
    cos=np.linalg.norm(POI[test_idx,:]-POI,axis=1,keepdims=True)# Euclidean distance
    #print(cos.shape)
    cos[test_idx,0]=max(cos)
    dist_min=np.min(cos)
    #print('dist_min',dist_min)
    hole_1=np.where(cos==dist_min)
    hole_idx=random.randint(0,len(hole_1[0])-1)
    max_idx=hole_1[0][hole_idx]
    max_idx_list.append(max_idx) # satellite image pairs with maximum similarity (minimum distance)
max_idx_array=np.array(max_idx_list)
np.savetxt('corr_POI_1st_euclid.txt',max_idx_array,fmt='%i') # satellite image index with maximum POI similarity
    
