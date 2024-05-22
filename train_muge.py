import torch
from torch import nn
train_root="/data/zhoucaixia/workspace/UD_Edge/"
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from statistics import mode
sys.path.append(train_root)
import numpy as np
import clip
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
from models.model_cls import VGG16
matplotlib.use('Agg')
from data.data_loader_one_random_alpha_uncert import BSDS_RCFLoader
MODEL_NAME="models.sigma_logit_unetpp_alpha_ffthalf_feat"
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
parser.add_argument('--tmp', help='tmp folder', default='/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_')
parser.add_argument('--dataset', help='root folder of dataset', default='/data/zhoucaixia/Dataset/BSDS/')
parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp+"{}_testalpha_clipsum".format(MODEL_NAME[7:]))

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
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.sum(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

import collections
class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps
class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        device = "cuda" 
        self.clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        clip.model.convert_weights(self.clip_model)
        self.visual_encoder = CLIPVisualEncoder(self.clip_model)
        self.criterionCLIP= torch.nn.MSELoss(reduction='sum')
    def forward(self, sketches, targets,label_style):
        sketches_channel0 = (sketches - 0.48145466) / 0.26862954
        sketches_channel1 = (sketches - 0.4578275) / 0.26130258
        sketches_channel2 = (sketches - 0.40821073) / 0.27577711
        sketches = torch.cat([sketches_channel0, sketches_channel1, sketches_channel2], dim=1)
        # sketches = sketches.repeat(1,3,1,1)
        # targets = targets.repeat(1,3,1,1)
        targets_channel0 = (targets - 0.48145466) / 0.26862954
        targets_channel1 = (targets - 0.4578275) / 0.26130258
        targets_channel2 = (targets - 0.40821073) / 0.27577711
        targets = torch.cat([targets_channel0, targets_channel1, targets_channel2], dim=1)
        
        sketches = torch.nn.functional.interpolate(sketches, size=224)
        targets = torch.nn.functional.interpolate(targets, size=224)
        
        targets_features = self.visual_encoder(targets)
        sketch_features = self.visual_encoder(sketches)
        loss_clip = self.criterionCLIP(sketch_features[0], targets_features[0].detach())
        
        return loss_clip
   
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
    
    
    cost = F.binary_cross_entropy(
                prediction, labelef, weight=mask.detach(), reduction='sum')
     
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
        num_workers=4, drop_last=True,shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=1, drop_last=True,shuffle=False)
    with open('/data/zhoucaixia/Dataset/BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model=Model.Mymodel(args).cuda()
    vgg_model=VGG16().cuda()
    trained_path='/data/zhoucaixia/workspace/UD_Edge/tmp/best_test_saved_weights.pth'
    vgg_model.load_state_dict(torch.load(trained_path))
    
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('Adam',args.LR)))
    sys.stdout = log
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.maxepoch):
        train(train_loader, model,vgg_model, optimizer,epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        log.flush() # write log


def train(train_loader, model,vgg_model,optimizer,epoch, save_dir):
    optimizer=step_lr_scheduler(optimizer,epoch)
    
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    vgg_model.eval()
    print(epoch,optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time()
    epoch_loss = []
    ffl=FocalFrequencyLoss()
    style=CLIPLoss()
    for i, (image, label,label_mean,label_std,_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label,label_std= image.cuda(), label.cuda(),label_std.cuda()
        label_style=vgg_model(torch.cat([image,label],1)).view(-1)
        
        mean,std= model(image,label_style.detach())
        outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
        outputs=torch.sigmoid(outputs_dist.rsample())
        
        bce_loss,mask=cross_entropy_loss_RCF(outputs,label)
        std_loss=torch.sum((std-label_std)**2*mask)
        ffl_loss=ffl(outputs,label)
        style_loss=style(outputs,label,label_style)
        loss = (bce_loss+std_loss+ffl_loss+style_loss) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
            print(bce_loss.item(),std_loss.item(),ffl_loss.item(),style_loss.item())
            print(label_style)
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
        for l in range(2):
            label_bias=torch.ones(1).cuda()
            label_bias=label_bias*l
            mean,std= model(image,label_bias)
            
            outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
            outputs=torch.sigmoid(outputs_dist.rsample())
            png=torch.squeeze(outputs.detach()).cpu().numpy()
            _, _, H, W = image.shape
            result=np.zeros((H+1,W+1))
            result[1:,1:]=png
            filename = splitext(test_list[idx])[0]
            result_png = Image.fromarray((result * 255).astype(np.uint8))
            
            png_save_dir=os.path.join(save_dir,str(l),"png")
            mat_save_dir=os.path.join(save_dir,str(l),"mat")

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
            mean_save_dir=os.path.join(save_dir,str(l),"mean")
            
            if not os.path.exists(mean_save_dir):
                os.makedirs(mean_save_dir)
            result_mean_png .save(join(mean_save_dir, "%s.png" % filename))

            std=torch.squeeze(std.detach()).cpu().numpy()
            result_std=np.zeros((H+1,W+1))
            result_std[1:,1:]=std
            result_std_png = Image.fromarray((result_std * 255).astype(np.uint8))
            std_save_dir=os.path.join(save_dir,str(l),"std")
            
            if not os.path.exists(std_save_dir):
                os.makedirs(std_save_dir)
            result_std_png .save(join(std_save_dir, "%s.png" % filename))

if __name__ == '__main__':
    main()
