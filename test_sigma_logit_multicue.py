import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

sys.path.append("/data/private/zhoucaixia/workspace/UD_Edge")
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
from data.data_loader_one_random_multicue_uncert import BSDS_RCFLoader
from models.sigma_logit_unetpp import Mymodel
from torch.utils.data import DataLoader
from torch.distributions import Normal, Independent
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
parser.add_argument('--fold', default='3', help='fold')
parser.add_argument('--type', help='edges or boundaries', default='edges')

args = parser.parse_args()

test_dataset = BSDS_RCFLoader(args,root='/data/private/zhoucaixia/Dataset/multicue/',  split= "test")
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=1, drop_last=True,shuffle=False)

with open(join(args.dataset, 'test_{}_all_{}.lst'.format(args.type,args.fold)), 'r') as f:
        test_list = f.readlines()
test_list = [split(i.rstrip())[1] for i in test_list]
assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

model_path="/data/private/zhoucaixia/workspace/UD_Edge/tmp_2023/multicue_sigma_logit_unetpp_fold3/epoch-2-training-record/epoch-2-checkpoint.pth"
model = Mymodel(args)

model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
save_dir = "/data/private/zhoucaixia/workspace/UD_Edge/tmp_2023/multicue_sigma_logit_unetpp_fold3_test/"

for idx, image in enumerate(test_loader):
    line=test_list[idx].strip("\n").split("\t")
    image = image.cuda()
    with torch.no_grad():
        
        batch_size, _, h_img, w_img = image.size()
        h_crop = 512
        w_crop =512
        h_stride = 200
        w_stride = 400
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = image.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = image.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = image[:, :, y1:y2, x1:x2]
                with torch.no_grad():
                    mean,std= model(crop_img)
                    outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
                    crop_seg_logit = torch.sigmoid(outputs_dist.rsample())
                preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        result=torch.squeeze(preds.detach()).cpu().numpy()
        filename = splitext(test_list[idx])[0]
        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir=os.path.join(save_dir,str(l),"png")
        mat_save_dir=os.path.join(save_dir,str(l),"mat")
        os.makedirs(png_save_dir,exist_ok=True)
        os.makedirs(mat_save_dir,exist_ok=True)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)
