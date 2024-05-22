import os, sys
sys.path.append("/data/private/zhoucaixia/workspace/UD_Edge")
import numpy as np
from PIL import Image
import cv2
import argparse
import time
import torch
import matplotlib
matplotlib.use('Agg')
from data.data_loader_one_random_drx import BSDS_RCFLoader
from models.sigma_logit_unetpp import Mymodel
from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from torch.distributions import Normal, Independent
from shutil import copyfile
from PIL import Image
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

test_dataset = BSDS_RCFLoader(root='/data/private/zhoucaixia/Dataset/data',  split= "test")
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=1, drop_last=True,shuffle=False)


with open('/data/private/zhoucaixia/Dataset/HED-BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
test_list = [split(i.rstrip())[1] for i in test_list]
assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

model_path="/data/private/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_gs_weightedstd1_declr_adaexp/epoch-19-training-record/epoch-19-checkpoint.pth"
model = Mymodel(args)

model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

save_dir = "/data/private/zhoucaixia/workspace/UD_Edge/test/UAED"
if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

for idx, image in enumerate(test_loader):
    filename = splitext(test_list[idx])[0]
    image = image.cuda()
    mean,std= model(image)
    _, _, H, W = image.shape
    outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
    outputs = torch.sigmoid(outputs_dist.rsample())
    png=torch.squeeze(outputs.detach()).cpu().numpy()
    result=np.zeros((H+1,W+1))
    result[1:,1:]=png
    result_png = Image.fromarray((result * 255).astype(np.uint8))
    png_save_dir=os.path.join(save_dir,"png")
    mat_save_dir=os.path.join(save_dir,"mat")
    os.makedirs(png_save_dir,exist_ok=True)
    os.makedirs(mat_save_dir,exist_ok=True)
    result_png.save(join(png_save_dir, "%s.png" % filename))
    io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)
