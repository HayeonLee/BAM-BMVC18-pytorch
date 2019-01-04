import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



# modified version of 
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# for cifar 100 (32x32)

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
    

def avg_norm_over_channel_axis(out, keepdim=True):
    out_mean = torch.mean(out, 1, keepdim=keepdim) #(B,C,H,W) -> (B,1,H,W)
    out_max = torch.max(torch.max(out_mean, 2, keepdim=True)[0], 3, keepdim=True)[0] # (B,1,1,1)
    out_min = torch.min(torch.min(out_mean, 2, keepdim=True)[0], 3, keepdim=True)[0] # (B,1,1,1)
    out_norm = (out_mean - out_min) / (out_max - out_min) # (B,1,H,W)
    
    return out_norm


#def avg_over_channel_axis(out, keepdim=True):
#    out_mean = torch.mean(out, 1, keepdim=keepdim) #(B,C,H,W) -> (B,1,H,W)
#    return out_mean
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class Restore_Image(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)
 
    
class Channel_Atten(nn.Module):
    
    def __init__(self, planes):
        super(Channel_Atten, self).__init__()

        self.AvgPool = nn.AdaptiveAvgPool2d((1,1))    
        self.fc1 = nn.Linear(planes * 4, planes * 4 / 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(planes * 4 / 16, planes * 4)
        self.at_sigmoid = nn.Sigmoid()
       
    def forward(self, x):
        ch_out = self.AvgPool(x)
        ch_out = ch_out.view(ch_out.size(0), -1)
        ch_out = self.fc1(ch_out)
        ch_out = self.relu(ch_out)
        ch_out = self.fc2(ch_out)
        ch_out = ch_out.view(ch_out.size(0), -1, 1, 1)        
        ch_out = self.at_sigmoid(ch_out)
        # channel-wise scale (B,C,H,W) * (B,C,1,1)
        x_at = x.clone() * ch_out
        x_at += x  
        
        return x_at

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
   
    
class ResNet_Atten(nn.Module):

    def __init__(self, block, layers, attention=None, vis_featmaps=False, num_classes=100):
        super(ResNet_Atten, self).__init__()
        self.inplanes = 64
        self.vis_featmaps = vis_featmaps
        # modified conv1
        # (3, 64, 64) -> (64, 64, 64), why no bias?
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #skip maxpool: 
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #(Bottleneck, 64, 3)
        self.atten1 = attention(64)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #(Bottleneck, 128, 4)
        self.atten2 = attention(128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.atten3 = attention(256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4, stride=1)
        self.fc100 = nn.Linear(512*block.expansion, num_classes)
        
        # To visualize heatmaps
        if self.vis_featmaps:
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')


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
        for i in range(1, num_blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, stride=1, channel_expansion=None))
        
        return nn.Sequential(*layers)
 
    def forward(self, x):
        images = []
        
        if self.vis_featmaps: # Before atten 1
            images.append(x) # (B, 3, 32, 32)
        
        x = self.conv_1(x) #(3, 32, 32) -> (64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        
        # (64,32,32) -> Bottleneck(64, 64, 1, channel_expansion): (256, 32, 32)
        # -> Bottleneck(256, 64, 1): (256, 32, 32) 
        # -> Bottleneck(256, 64, 1): (256, 32, 32)
        x = self.layer1(x)  ## 32
        if self.vis_featmaps: # Before atten 1
            images.append(torch.squeeze(self.upsample1(avg_norm_over_channel_axis(x))))
            #images.append(torch.squeeze(avg_norm_over_channel_axis(x)))

        x = self.atten1(x)
        if self.vis_featmaps: # After atten 1
#            images.append(torch.squeeze(avg_norm_over_channel_axis(x)))
            images.append(torch.squeeze(self.upsample1(avg_norm_over_channel_axis(x))))

        x = self.layer2(x) #(256, 32, 32) -> (128, 16, 16) -> (512, 16, 16) ##16
        if self.vis_featmaps:# Before atten 2
            images.append(torch.squeeze(self.upsample2(avg_norm_over_channel_axis(x))))
            #images.append(torch.squeeze(avg_norm_over_channel_axis(x)))

        x = self.atten2(x)
        if self.vis_featmaps: # After atten 2
            images.append(torch.squeeze(self.upsample2(avg_norm_over_channel_axis(x))))
            #images.append(torch.squeeze(avg_norm_over_channel_axis(x)))

        x = self.layer3(x) #(512, 16, 16) -> (256, 8, 8) -> (1024, 8, 8) ##8
        if self.vis_featmaps: # Before atten 3
            images.append(torch.squeeze(self.upsample3(avg_norm_over_channel_axis(x))))
            #images.append(torch.squeeze(avg_norm_over_channel_axis(x)))

        x = self.atten3(x)
        if self.vis_featmaps:
            images.append(torch.squeeze(self.upsample3(avg_norm_over_channel_axis(x))))
            #images.append(torch.squeeze(avg_norm_over_channel_axis(x)))

        x = self.layer4(x) #(1024, 8, 8) -> (512, 4, 4) -> (2048, 4, 4) ##4
        
        x = self.avg_pool(x) #(2048, 1, 1)
        x = x.view(x.size(0), -1) # (2048, )
        x = self.fc100(x)
        images.append(x)
        return images
        #return x

    
def resnet50_channel_atten(pretrain=True, vis_featmaps=False):
    model = ResNet_Atten(Bottleneck, [3, 4, 6, 3], Channel_Atten, vis_featmaps)
    if pretrain:
        load_pretrained_weight(model)

    return model


# def resnet50_spatial_atten(pretrain=True, vis_featmaps=False):
#     model = ResNet_Atten(Bottleneck, [3, 4, 6, 3], Spatial_Atten, vis_featmaps)
#     if pretrain:
#         load_pretrained_weight(model)

#     return model


# def resnet50_joint_atten(pretrain=True, vis_featmaps=False):
#     model = ResNet_Atten(Bottleneck, [3, 4, 6, 3], Joint_Atten, vis_featmaps)
#     if pretrain:
#         load_pretrained_weight(model)

#     return model

