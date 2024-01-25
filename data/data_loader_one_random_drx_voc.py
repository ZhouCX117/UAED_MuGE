from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random
import cv2
train_root='/data/private/zhoucaixia/Dataset/HED-BSDS/PASCAL'
test_root='/data/private/zhoucaixia/Dataset/HED-BSDS'
class VOC_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(train_root, 'train_pair.lst')
            
        elif self.split == 'test':
            self.filelist = join(test_root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        self.train_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(384),
                
            ]
        )
        self.test_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            lb_file=img_lb_file[-1]
            img = imageio.imread(join(train_root,img_file))
            img=self.train_transform(img)
            img=transforms.ToTensor()(img)
        else:
            img_file = self.filelist[index].rstrip()
            img = imageio.imread(join(test_root,img_file))
            img=img[1:,1:,:]
            img=self.test_transform(img)
        
        img = img.float()

        
        if self.split == "train":
            
            lb = scipy.io.loadmat(join(self.root,lb_file))
            label=np.asarray(lb['edge_gt']).astype(np.uint8)
            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            label=self.train_transform(label)
            label=np.array(label)
            label=torch.from_numpy(label)
            label[label > 0] = 1
            label=label.unsqueeze(0)
            label = label.float()
                
            return img, label
            #,init.permute(2,0,1)
        else:
            return img
            #,init.permute(2,0,1)

