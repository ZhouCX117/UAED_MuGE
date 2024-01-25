from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random
from PIL import Image
import torchvision.transforms.functional as F
class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, args,root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train_{}_all_{}.lst'.format(args.type,args.fold))
            
        elif self.split == 'test':
            self.filelist = join(self.root, 'test_{}_all_{}.lst'.format(args.type,args.fold))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def crop_500(self, img, gt_list):
        w, h = img.size
        random_w = np.random.randint(0, w-512)
        random_h = np.random.randint(0, h-512)

        box = (random_w, random_h, random_w+512, random_h+512)
        img = img.crop(box)
        for index in range(len(gt_list)):
            gt_list[index] = gt_list[index].crop(box) 
            gt_list[index] = torch.from_numpy(np.array(gt_list[index]))
            gt_list[index] = gt_list[index].unsqueeze(0)
            gt_list[index] = gt_list[index].float() # if crop 500 + multi-scale to train, the 1280x720 is able to train

        img = transforms.ToTensor()(img)
        img = img.float()

        return img,gt_list

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            label_list=[]
            for i_label in range(1,len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root,img_lb_file[i_label]))
                lb=np.asarray(lb['edge_gt'])
                lb=Image.fromarray(lb)
                label_list.append(lb)
        else:
            img_file = self.filelist[index].strip("\n").split(" ")[0]

        # img = imageio.imread(join(self.root,img_file))
        img=Image.open(join(self.root,img_file)).convert('RGB')
        
        if self.split == "train":
            
            img,label_list=self.crop_500(img,label_list)
            labels=torch.cat(label_list,0)
            lb_mean=labels.mean(dim=0).unsqueeze(0)
            lb_std=labels.std(dim=0).unsqueeze(0)

            lb_index=random.randint(1,len(label_list))-1
            label=label_list[lb_index]
            return img, label,lb_mean,lb_std
            #,init.permute(2,0,1)
        else:
            img = transforms.ToTensor()(img)
            img=img.float()
            return img
            #,init.permute(2,0,1)

