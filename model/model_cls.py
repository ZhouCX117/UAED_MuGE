import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self,num_classes=1):
        super(VGG16, self).__init__()
        #导入VGG16模型
        model = models.vgg16_bn(pretrained=True)
        #加载features部分
        self.features = model.features
        self.features[0]=nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
       
        #加载avgpool层
        self.avgpool=model.avgpool
        #改变classifier：分类输出层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512*7*7)
        logits=self.classifier(x)
        logits = torch.sigmoid(logits)
        return logits
