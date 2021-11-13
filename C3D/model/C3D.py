"""
@author:  Tongjia (Tom) Chen
@contact: tomchen@hnu.edu.cn
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class C3D(nn.Module):
    def __init__(self, num_classes=101, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1a', nn.Conv3d(3, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('pool1', nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))),
            ('conv2a', nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('pool2', nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))),
            ('conv3a', nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('conv3b', nn.Conv3d(256, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('pool3', nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))),
            ('conv4a', nn.Conv3d(256, 512, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('conv4b', nn.Conv3d(512, 512, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('pool4', nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))),
            ('conv5a', nn.Conv3d(512, 512, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('conv5b', nn.Conv3d(512, 512, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            ('relu', nn.ReLU()),
            ('pool5', nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))
        ]))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv(x)
        x = x.view(-1, 8192)
        x = self.fc6(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = self.dropout(x)
        x = F.relu(x)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    from IPython import embed
    model = C3D()
    x = torch.Tensor(8, 3, 16, 112, 112)
    y = model(x)
    embed()
    import cv2
    cv2.norm()