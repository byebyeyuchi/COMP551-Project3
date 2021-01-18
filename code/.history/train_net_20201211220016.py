
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import argparse
from time import time
from load_data import DataLoader_v2
from torch.utils.data import DataLoader
from module_list.resnet_v1 import resnet50,resnet18,resnet34


parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet18' , help='net type')  
#parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')  # whether use gpu or not
parser.add_argument('-bz', type=int, default=32, help='batch size')          # chose Batchsize
parser.add_argument('-epoch', type=int, default=30, help='epoch')        # choose Epoch 训练轮数
parser.add_argument('-w', type=int, default=0, help='num_workers')
parser.add_argument('-output', type=str, default='net1', help='checkpoint dir') 
parser.add_argument('-pre', type=bool, default=False, help='if use pretrained model')
parser.add_argument('-data_aug', type=bool, default=True, help='if use data augmentation')
parser.add_argument('-lr', type=float, default=0.0001, help=' learning rate')
args = parser.parse_args()


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)  #设置随机种子
torch.cuda.manual_seed_all(1234)
os.environ['PYTHONHASHSEED'] = str(1234)
torch.backends.cudnn.deterministic = True

use_gpu = torch.cuda.is_available()
print (use_gpu)

# store model
batchSize = args.bz
modelFolder = 'output/' + args.output + '/'
storeName = modelFolder + 'net.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)
epochs = args.epoch



if args.net == 'resnet50':
    model = resnet50()
    
elif args.net == 'resnet18':
    model = resnet18()

elif args.net == 'resnet34':
    model = resnet34()
    

    

model_conv = model
if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()


optimizer_conv = optim.Adam( model_conv.parameters(), lr=args.lr, betas=(0.9, 0.999))
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_conv, milestones=[epochs//3,2*epochs//3], gamma=0.2) 



dst = DataLoader_v2('data/train', 'data/labels')
dst_val = DataLoader_v2('data/val', 'data/labels')

# dst_test = DataLoader_v2('data/test', 'data/labels')


trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=args.w)
# test_dataloader = DataLoader(dst_test, batch_size=1, shuffle=False, num_workers=args.w)
val_dataloader = DataLoader(dst_val, batch_size=batchSize//2, shuffle=True, num_workers=args.w)

draw_train_acc = []
draw_train_loss = []
draw_test_acc = []
draw_test_loss = []

def train_model(model, criterion, optimizer, num_epochs=100):
    # since = time.time()
    max_acc=0
    for epoch in range( num_epochs):
        lossAver = []
        model.train()
        start = time()

        for i, (XI, y1, y2, y3,y4,y5) in enumerate(trainloader):

            if use_gpu:
                x = XI.cuda()
                y1 = y1.long().cuda()
                y2 = y2.long().cuda()
                y3 = y3.long().cuda()
                y4 = y4.long().cuda()
                y5 = y5.long().cuda()

            else:
                x = XI
                y1 = y1.long()
                y2 = y2.long()
                y3 = y3.long()
                y4 = y4.long()
                y5 = y5.long()
                
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = model(x)

            loss = 0.0
            
            loss1 =criterion(y_pred_1, y1)
            loss2 =criterion(y_pred_2, y2)
            loss3 =criterion(y_pred_3, y3)
            loss4 =criterion(y_pred_4, y4)
            loss5 =criterion(y_pred_5, y5)

            loss  =  loss1 + loss2+ loss3+ loss4+ loss5
            loss.requires_grad_(True)
     
            lossAver.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('epoch:%s  loss:%s  cost time:%s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start),'LR:%.6f'%optimizer.param_groups[0]['lr'])        
        Loss = sum(lossAver) / len(lossAver)
        draw_train_loss.append(round(float(Loss),4))
        
        model.eval()
        correct=0
        test_lossAver =[]
        for i, (XI, y1, y2, y3,y4,y5) in enumerate(val_dataloader):

            if use_gpu:
                x = XI.cuda()
                y1 = y1.long().cuda()
                y2 = y2.long().cuda()
                y3 = y3.long().cuda()
                y4 = y4.long().cuda()
                y5 = y5.long().cuda()
    
            else:
                x = XI
                y1 = y1.long()
                y2 = y2.long()
                y3 = y3.long()
                y4 = y4.long()
                y5 = y5.long()
                               
            y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = model(x)
                
            loss1 =criterion(y_pred_1, y1)
            loss2 =criterion(y_pred_2, y2)
            loss3 =criterion(y_pred_3, y3)
            loss4 =criterion(y_pred_4, y4)
            loss5 =criterion(y_pred_5, y5)
    
            loss  =  loss1  + loss2+ loss3+ loss4+ loss5
            test_lossAver.append(loss.item()) 
                
            predicted_1 = torch.max(y_pred_1, 1)[1].data.squeeze().cpu().numpy()  
            predicted_2 = torch.max(y_pred_2, 1)[1].data.squeeze().cpu().numpy()  
            predicted_3 = torch.max(y_pred_3, 1)[1].data.squeeze().cpu().numpy()  
            predicted_4 = torch.max(y_pred_4, 1)[1].data.squeeze().cpu().numpy()  
            predicted_5 = torch.max(y_pred_5, 1)[1].data.squeeze().cpu().numpy()  
            
            result1 = predicted_1 == y1.data.cpu().numpy()
            result2 = predicted_2 == y2.data.cpu().numpy()
            result3 = predicted_3 == y3.data.cpu().numpy()
            result4 = predicted_4 == y4.data.cpu().numpy()
            result5 = predicted_5 == y5.data.cpu().numpy()
           
            correct = correct + np.sum(result1&result2&result3&result4&result5)
            
        ACC = round(100*correct / len(dst_val.img_paths) ,3)
        print ('###### val set loss:%s  ACC:%s  #####\n' 
                % ( sum(test_lossAver) / len(test_lossAver),ACC) )    
        
        if (epoch+1) % 20 == 0 or epoch==10 :
            torch.save(model.state_dict(), storeName + str(epoch+1))
        
        if ACC >= max_acc:
            torch.save(model.state_dict(), storeName + 'best')
            max_acc = ACC

        train_scheduler.step()
       
    print('max_acc:',max_acc,'\n') 

    return model




model_conv1 = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
