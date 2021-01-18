from __future__ import print_function, division
import cv2
import torch
import numpy as np
import os
import argparse
from module_list.resnet_v1 import resnet50,resnet18,resnet34
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet50' , help='net type')  # define your network here
parser.add_argument('-output', type=str, default='resnet50', help='checkpoint dir') 
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print (use_gpu)

modelFolder = 'output/' + args.output + '/'
storeName = modelFolder + 'time_net.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)


if args.net == 'resnet50':
    model = resnet50()
    
if args.net == 'resnet18':
    model = resnet18()

elif args.net == 'resnet34':
    model = resnet34()

model.load_state_dict(torch.load(storeName + 'best'))

model.eval()

if use_gpu:
    model = model.cuda()


def process_img(img_name):
    img = cv2.imread(img_name,0)
    img = img[:,:,np.newaxis]
    img= torch.from_numpy(img.astype(np.float32)).permute(2,0,1)

    img /= 255.0
    return img

id_list=['Id']
label_list = ['Label']
for i in range(14000):
    imgpath = os.path.join('data', 'test', str(i).zfill(5)+'.png')
    XI = process_img(imgpath)
    XI = XI.unsqueeze(0).cuda()
    y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5  = model(XI)
    
    predicted_1 = torch.max(y_pred_1, 1)[1].data.squeeze().cpu().numpy()  
    predicted_2 = torch.max(y_pred_2, 1)[1].data.squeeze().cpu().numpy()  
    predicted_3 = torch.max(y_pred_3, 1)[1].data.squeeze().cpu().numpy()  
    predicted_4 = torch.max(y_pred_4, 1)[1].data.squeeze().cpu().numpy()  
    predicted_5 = torch.max(y_pred_5, 1)[1].data.squeeze().cpu().numpy()  
    result = str(predicted_1)+str(predicted_2)+str(predicted_3)+str(predicted_4)+str(predicted_5)
    # print(result)
    id_list.append(str(i))
    label_list.append(result)
    
with open('result.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(id_list,label_list))














