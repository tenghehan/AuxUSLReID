from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50']

class ResNet50(nn.Module):

    def __init__(self, num_classes, loss={'xent', 'htri'}, **kwargs):
        super(ResNet50, self).__init__()
        # Construct base (pretrained) resnet
        resnet = torchvision.models.resnet50(pretrained=True)
        # resnet.layer4[0].conv2.stride = (1,1)
        # resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.num_classes = num_classes
        out_planes = resnet.fc.in_features

        self.loss = loss

        # Change the num_features to CNN output channels
        self.num_features = out_planes
        assert self.num_features == 2048
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])

        bn_x = x.view(x.size(0), -1)
        bn_x = self.feat_bn(bn_x)

        bn_x_norm = F.normalize(bn_x)
        
        bn_x_norm = bn_x_norm.view(b,t,-1)
        bn_x_norm = bn_x_norm.permute(0,2,1)
        bn_f_norm = F.avg_pool1d(bn_x_norm,t)
        bn_f_norm = bn_f_norm.view(b, self.num_features)

        bn_x = bn_x.view(b,t,-1)
        bn_x = bn_x.permute(0,2,1)
        bn_f = F.avg_pool1d(bn_x,t)
        bn_f = bn_f.view(b, self.num_features)

        x = x.view(b,t,-1)
        x = x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.num_features)

        if self.training is False:
            return bn_f_norm, f
            # return bn_f_norm, bn_f_norm
            # return bn_f_norm, F.normalize(f)

        y = self.classifier(bn_f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return f, y
        elif self.loss == {'cent'}:
            return f, y
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


