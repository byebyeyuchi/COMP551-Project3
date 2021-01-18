from torch.utils.data import Dataset
from imutils import paths
import cv2
import numpy as np
import json
import os
import torch


class DataLoader_v2(Dataset):
    def __init__(self, img_dir, label_dir,is_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = []   
        self.img_paths += [el for el in paths.list_images(img_dir)]   
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        label_name = os.path.join(self.label_dir,img_name[-9:-4]+'.json')    
        with open(label_name, 'r') as f:
            label = json.load(f)
        
        img = cv2.imread(img_name,0)
        img = img[:,:,np.newaxis]
        resizedImage= torch.from_numpy(img.astype(np.float32)).permute(2,0,1)
 
        label1 = float(label[0])
        label2 = float(label[1])
        label3 = float(label[2])
        label4 = float(label[3])
        label5 = float(label[4])
        
        resizedImage /= 255.0
       
        return resizedImage , label1, label2, label3, label4, label5
