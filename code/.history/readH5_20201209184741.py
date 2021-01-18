# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:26:41 2020

@author: wuxiaoliang
"""

import h5py
import os
import cv2
import json

filename = 'MNIST_synthetic.h5'
f = h5py.File(filename, 'r')



train_set = f['train_dataset']
train_label = f['train_labels']
test_set = f['test_dataset']


for i in range(len(train_set)):
    
    if i <50000:
        imgpath = os.path.join('data', 'train', str(i).zfill(5)+'.png')
    else:
        imgpath = os.path.join('data', 'val', str(i).zfill(5)+'.png')
    cv2.imwrite(imgpath, train_set[i])


for i in range(len(train_label)):
    label_path = os.path.join('data', 'labels', str(i).zfill(5)+'.json')
    label = train_label[i].tolist()
    with open(label_path,'w') as file_obj:
        json.dump(label,file_obj)

    
for i in range(len(test_set)):
    imgpath = os.path.join('data', 'test', str(i).zfill(5)+'.png')
    cv2.imwrite(imgpath, test_set[i])
