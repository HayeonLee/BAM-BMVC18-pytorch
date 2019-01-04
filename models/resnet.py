import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from visdom import Visdom
import math

# modified version of 
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# for cifar 100 (32x32

#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152']

model_urls = {
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            }

def load_pretrained_weight(model):
    print('    Pretrain the model using imageNet...')
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()
    #print(model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    
class Bottleneck(nn.Module):
    expansion = 4

    
    def __init__(self, inplanes, planes, stride=1, channel_expansion=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.channel_expansion = channel_expansion
        self.stride = stride
       
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.channel_expansion is not None:
            residual = self.channel_expansion(residual)
            
        out += residual
        out = self.relu(out)
        
        return out

    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # modified conv1
        # (3, 64, 64) -> (64, 64, 64), why no bias?
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #skip maxpool: 
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #(Bottleneck, 128, 4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4, stride=1)
        self.fc100 = nn.Linear(512*block.expansion, num_classes)
        self.vis = Visdom()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                #nn.init.kaiming_uniform(m.weight.data)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)
        '''


    def _make_layer(self, block, planes, num_blocks, stride=1):
        channel_expansion = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # channel_expansion
            # (64, ?, ?) -> (256, ?, ?)
            channel_expansion = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, channel_expansion))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        images = []
        x = self.conv_1(x) #(3, 32, 32) -> (64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        
        # (64,32,32) -> Bottleneck(64, 64, 1, channel_expansion): (256, 32, 32)
        # -> Bottleneck(256, 64, 1): (256, 32, 32) 
        # -> Bottleneck(256, 64, 1): (256, 32, 32)
        x = self.layer1(x) #(B, 256, 32, 32)
        x = self.layer2(x) #(B,256, 32, 32) -> (B,128, 16, 16) -> (B,512, 16, 16)
        x = self.layer3(x) #(B,512, 16, 16) -> (B,256, 8, 8) -> (B,1024, 8, 8)
        x = self.layer4(x) #(B,1024, 8, 8) -> (B,512, 4, 4) -> (B,2048, 4, 4)
        x = self.avg_pool(x) #(B,2048, 1, 1)
        x = x.view(x.size(0), -1) # (2048, )
        x = self.fc100(x)
        
        images.append(x)
        return images
    

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_pretrained_weight(model)
    
    return model



