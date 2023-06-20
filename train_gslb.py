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

from data.data_loader_one_random_drx_uncert import BSDS_RCFLoader
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
import cv2
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
parser.add_argument('--gpu', default='7', type=str,
                    help='GPU ID')
parser.add_argument('--tmp', help='tmp folder', default='/data/private/zhoucaixia/workspace/UD_Edge/tmp/trainval_')
parser.add_argument('--dataset', help='root folder of dataset', default='/data/private/zhoucaixia/Dataset/data/')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--std_weight', default=1, type=float,help='weight for std loss')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp+"{}_{}_weightedstd{}_declr_adaexp".format(MODEL_NAME[7:],args.distribution,args.std_weight))


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
 
def cross_entropy_loss_RCF(prediction, labelef,std,ada):
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
    
    new_mask=mask*torch.exp(std*ada)
    cost = F.binary_cross_entropy(
                prediction, labelef, weight=new_mask.detach(), reduction='sum')
     
    return cost,mask
def step_lr_scheduler(optimizer, epoch, init_lr=args.LR, lr_decay_epoch=3):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
def main():
    args.cuda = True
    train_dataset = BSDS_RCFLoader(root=args.dataset, split= "train")
    test_dataset = BSDS_RCFLoader(root=args.dataset,  split= "test")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=8, drop_last=True,shuffle=False)
    with open('/data/private/zhoucaixia/Dataset/HED-BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model=Model.Mymodel(args).cuda()

    

    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('Adam',args.LR)))
    sys.stdout = log
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.maxepoch):
        # if epoch==0:
        #     test(model, test_loader, epoch=epoch, test_list=test_list,
        #     save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        train(train_loader, model, optimizer,epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush() # write log


def train(train_loader, model,optimizer,epoch, save_dir):
    optimizer=step_lr_scheduler(optimizer,epoch)
    
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    print(epoch,optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label,label_mean,label_std) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label,label_std = image.cuda(), label.cuda(),label_std.cuda()
        
        mean,std= model(image)
        
        outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)

        outputs=torch.sigmoid(outputs_dist.rsample())
        counter += 1
        
        ada=(epoch+1)/args.maxepoch
        bce_loss,mask=cross_entropy_loss_RCF(outputs,label,std,ada)

        
        std_loss=torch.sum((std-label_std)**2*mask)
            
    
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
            torchvision.utils.save_image(1-mean, join(save_dir, "iter-%d_mean.jpg" % i))
            torchvision.utils.save_image(1-std, join(save_dir, "iter-%d_std.jpg" % i))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))



def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        mean,std= model(image)
        outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
        outputs=torch.sigmoid(outputs_dist.rsample())
        png=torch.squeeze(outputs.detach()).cpu().numpy()
        _, _, H, W = image.shape
        result=np.zeros((H+1,W+1))
        result[1:,1:]=png
        filename = splitext(test_list[idx])[0]
        result_png = Image.fromarray((result * 255).astype(np.uint8))
        
        png_save_dir=os.path.join(save_dir,"png")
        mat_save_dir=os.path.join(save_dir,"mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)

        mean=torch.squeeze(mean.detach()).cpu().numpy()
        result_mean=np.zeros((H+1,W+1))
        result_mean[1:,1:]=mean
        result_mean_png = Image.fromarray((result_mean).astype(np.uint8))
        mean_save_dir=os.path.join(save_dir,"mean")
        
        if not os.path.exists(mean_save_dir):
            os.makedirs(mean_save_dir)
        result_mean_png .save(join(mean_save_dir, "%s.png" % filename))

        std=torch.squeeze(std.detach()).cpu().numpy()
        result_std=np.zeros((H+1,W+1))
        result_std[1:,1:]=std
        result_std_png = Image.fromarray((result_std * 255).astype(np.uint8))
        std_save_dir=os.path.join(save_dir,"std")
        
        if not os.path.exists(std_save_dir):
            os.makedirs(std_save_dir)
        result_std_png .save(join(std_save_dir, "%s.png" % filename))

def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.6, 1, 1.6]
    for idx, image in enumerate(test_loader):
        image = image[0]
        image_in = image.numpy().transpose((1,2,0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2,0,1))

            mean,std= model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
            outputs=torch.sigmoid(outputs_dist.rsample())
            result = torch.squeeze(outputs.detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        
        result=np.zeros((H+1,W+1))
        result[1:,1:]=multi_fuse
        filename = splitext(test_list[idx])[0]

        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir=os.path.join(save_dir,"png")
        mat_save_dir=os.path.join(save_dir,"mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)
if __name__ == '__main__':
    main()
   
