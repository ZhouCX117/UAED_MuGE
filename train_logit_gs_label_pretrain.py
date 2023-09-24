import torch
from torch import nn

#!/user/bin/python
# coding=utf-8
train_root="/data/private/zhoucaixia/workspace/UD_Edge/"
import os, sys
from statistics import mode
sys.path.append(train_root)

import numpy as np
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import cv2
from data.data_loader_one_random_drx_voc import VOC_RCFLoader
MODEL_NAME="models.sigma_logit_unetpp"
import importlib
Model = importlib.import_module(MODEL_NAME)

from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
import random
import numpy
from torch.autograd import Variable
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torch.distributions import Normal, Independent
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=4, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--LR', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--tmp', help='tmp folder', default='/data/private/zhoucaixia/workspace/UD_Edge/tmp/voc_pretrain_')
parser.add_argument('--dataset', help='root folder of dataset', default='/data/private/zhoucaixia/Dataset/HED-BSDS/PASCAL')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--std_weight', default=1, type=float,help='weight for std loss')
parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp+"{}_{}_weightedstd{}_labelstd".format(MODEL_NAME[7:],args.distribution,args.std_weight))

if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

file_name=os.path.basename(__file__)
copyfile(join(train_root,MODEL_NAME[:6],MODEL_NAME[7:]+".py"),join(TMP_DIR,MODEL_NAME[7:]+".py"))
copyfile(join(train_root,"train",file_name),join(TMP_DIR,file_name))
random_seed = 555
if random_seed > 0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)
 
def cross_entropy_loss_RCF(prediction, labelef):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()
    num_two=torch.sum((mask==2).float()).float()
    assert num_negative+num_positive+num_two==label.shape[0]*label.shape[1]*label.shape[2]*label.shape[3]
    assert num_two==0
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    #cost = F.binary_cross_entropy(
           #prediction, labelef, weight=mask, reduction='sum')
    
    cost = F.binary_cross_entropy(
                prediction, labelef, weight=mask, reduction='sum')
   
                
    return cost,mask  

def main():
    args.cuda = True
    train_dataset = VOC_RCFLoader(root=args.dataset, split= "train")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=True)
    
    # model
    model=Model.Mymodel(args).cuda()
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('Adam',args.LR)))
    sys.stdout = log
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.maxepoch):
        train(train_loader, model, optimizer,epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        log.flush() # write log


def train(train_loader, model,optimizer,epoch, save_dir):
    
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    print(epoch,optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label= image.cuda(), label.cuda()
        
        mean,std= model(image)
        outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
        outputs=torch.sigmoid(outputs_dist.rsample())
        
        counter += 1

        bce_loss,mask=cross_entropy_loss_RCF(outputs,label)
        
       
        std_loss=torch.sum((std-label)**2*mask)
    
        loss = (bce_loss+std_loss*args.std_weight) / args.itersize
        
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        losses.update(loss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            print(bce_loss.item(),std_loss.item())
            _, _, H, W = outputs.shape
            torchvision.utils.save_image(1-outputs, join(save_dir, "iter-%d.jpg" % i))
            torchvision.utils.save_image(1-std, join(save_dir, "iter-%d-std.jpg" % i))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))


if __name__ == '__main__':
    main()
   
