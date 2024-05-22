import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("/data/zhoucaixia/workspace/UD_Edge")
import numpy as np
from PIL import Image
import cv2
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from data.data_loader_one_random_drx import BSDS_RCFLoader
from models.sigma_logit_unetpp_alpha_ffthalf_feat import Mymodel
from torch.utils.data import DataLoader
from torch.distributions import Normal, Independent
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset', help='root folder of dataset', default='/data/zhoucaixia/Dataset/BSDS/')
parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')

args = parser.parse_args()

test_dataset = BSDS_RCFLoader(root='/data/zhoucaixia/Dataset/BSDS',  split= "test")
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=1, drop_last=True,shuffle=False)

with open('/data/zhoucaixia/Dataset/BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
test_list = [split(i.rstrip())[1] for i in test_list]
assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

model_path="/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/epoch-19-training-record/epoch-19-checkpoint.pth"
model = Mymodel(args)

model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
save_dir = "/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 
train_root="/data/zhoucaixia/workspace/UD_Edge/"
file_name=os.path.basename(__file__)
copyfile(join(train_root,"test",file_name),join(save_dir,file_name))

for idx, image in enumerate(test_loader):
    line=test_list[idx].strip("\n").split("\t")
    alpha_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
   
    image = image.cuda()
    with torch.no_grad():
        for l in alpha_list:
            label_bias=torch.ones(1).cuda()
            label_bias=label_bias*l
            mean,std= model(image,label_bias)
            outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
            outputs=torch.sigmoid(outputs_dist.rsample())
            
            png=torch.squeeze(outputs.detach()).cpu().numpy()
            _, _, H, W = image.shape
            result=np.zeros((H+1,W+1))
            result[1:,1:]=png
            filename = line[0].split("\t")[-1][:-4]
            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir=os.path.join(save_dir,str(l),"png")
            mat_save_dir=os.path.join(save_dir,str(l),"mat")
            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)
            if not os.path.exists(mat_save_dir):
                os.makedirs(mat_save_dir)
            print(png_save_dir)
            result_png.save(join(png_save_dir, "%s.png" % (filename)))
            io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)
