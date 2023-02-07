'''
Architecture Definition:


'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet
import torch.nn.functional as F

# GPU UTILIZATION
import GPUtil


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGGNet(VGG):
    def __init__(self, pretrained = True, model = 'vgg16', requires_grad = True, remove_fc = True, show_params = False):
        super().__init__(make_layers(cfg[model], batch_norm=False))
        self.ranges = ranges[model]  
        if pretrained:
            self.load_state_dict(models.vgg16(pretrained=True).state_dict())       
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False
        if remove_fc:
            del self.classifier
        if show_params:
            for name,param in self.named_parameters():
                print(name, param.size())   
    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        return output


class FCNDNC(nn.Module):
    def __init__(self, pretrained_net, n_class, p = 0.5):
        super().__init__()
        self.n_class = n_class
        self.compress = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=1)
        self.pretrained_net = pretrained_net
        # This one is for NL-Fusion w/o attention vector
        self.fusionConv = nn.ModuleDict({'x5': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),  
                           'x4': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                           'x3': nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                           'x2': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                           'x1': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)})

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class*2, kernel_size=1)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x_rgb, x_depth, x_normal, x_curvature):
        dnc = self.compress(torch.cat([x_depth,x_normal,x_curvature],1))
        output_rgb = self.pretrained_net(x_rgb)
        output_depth = self.pretrained_net(dnc)
        
        x5_rgb = output_rgb['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_rgb = output_rgb['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_rgb = output_rgb['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_rgb = output_rgb['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_rgb = output_rgb['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x5_depth = output_depth['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_depth = output_depth['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_depth = output_depth['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_depth = output_depth['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_depth = output_depth['x1']  # size=(N, 64, x.H/2,  x.W/2)

        # # # Fusion Block : NON-LINEAR WEIGHTED COMBINATION 
        x5 = self.relu(self.fuseBlock_1(x5_rgb, x5_depth, 'x5'))
        x4 = self.relu(self.fuseBlock_1(x4_rgb, x4_depth, 'x4'))
        x3 = self.relu(self.fuseBlock_1(x3_rgb, x3_depth, 'x3'))
        x2 = self.relu(self.fuseBlock_1(x2_rgb, x2_depth, 'x2'))
        x1 = self.relu(self.fuseBlock_1(x1_rgb, x1_depth, 'x1'))

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                              # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.dropout(score)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.dropout(score)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.dropout(score)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.dropout(score)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def fuseBlock_1(self, rgb, depth, layer):
        '''
        Fusion Block 1: 
        Weighted mean of RGB and Depth using 1x1 convolution 
        '''
        fused = torch.cat((rgb, depth), dim=1)
        x = self.fusionConv[layer](fused)
        return x

class FCNDC(nn.Module):
    def __init__(self, pretrained_net, n_class, p = 0.5):
        super().__init__()
        self.n_class = n_class
        self.compress = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1)
        self.pretrained_net = pretrained_net
        # self.aspp = ASPP(in_channels=512, out_channels=512, dilation_rates = [6,12,18])
        # This one is for NL-Fusion w/o attention vector
        self.fusionConv = nn.ModuleDict({'x5': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),  
                           'x4': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                           'x3': nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                           'x2': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                           'x1': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)})
        # self.fusionConv = nn.Conv2d(1024, 512, 1)

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class*2, kernel_size=1)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x_rgb, x_depth, x_curvature):
        dc = self.compress(torch.cat([x_depth,x_curvature],1))
        output_rgb = self.pretrained_net(x_rgb)
        output_depth = self.pretrained_net(dc)
        
        x5_rgb = output_rgb['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_rgb = output_rgb['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_rgb = output_rgb['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_rgb = output_rgb['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_rgb = output_rgb['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x5_depth = output_depth['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_depth = output_depth['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_depth = output_depth['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_depth = output_depth['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_depth = output_depth['x1']  # size=(N, 64, x.H/2,  x.W/2)

        # # # Fusion Block : NON-LINEAR WEIGHTED COMBINATION 
        x5 = self.relu(self.fuseBlock_1(x5_rgb, x5_depth, 'x5'))
        x4 = self.relu(self.fuseBlock_1(x4_rgb, x4_depth, 'x4'))
        x3 = self.relu(self.fuseBlock_1(x3_rgb, x3_depth, 'x3'))
        x2 = self.relu(self.fuseBlock_1(x2_rgb, x2_depth, 'x2'))
        x1 = self.relu(self.fuseBlock_1(x1_rgb, x1_depth, 'x1'))

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                              # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.dropout(score)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.dropout(score)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.dropout(score)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.dropout(score)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def fuseBlock_1(self, rgb, depth, layer):
        '''
        Fusion Block 1: 
        Weighted mean of RGB and Depth using 1x1 convolution 
        '''
        fused = torch.cat((rgb, depth), dim=1)
        # weight = F.softmax(self.fusionConv[layer].weight, dim=1)
        x = self.fusionConv[layer](fused)
        # x = F.conv2d(fused, weight)

        return x


class FCNDepth(nn.Module):
    def __init__(self, pretrained_net, n_class, p = 0.5):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        # self.aspp = ASPP(in_channels=512, out_channels=512, dilation_rates = [6,12,18])
        # This one is for NL-Fusion w/o attention vector
        self.fusionConv = nn.ModuleDict({'x5': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),  
                           'x4': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                           'x3': nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                           'x2': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                           'x1': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)})
        # self.fusionConv = nn.Conv2d(1024, 512, 1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class*2, kernel_size=1)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x_rgb, x_depth):
        output_rgb = self.pretrained_net(x_rgb)
        output_depth = self.pretrained_net(x_depth)
        
        x5_rgb = output_rgb['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_rgb = output_rgb['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_rgb = output_rgb['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_rgb = output_rgb['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_rgb = output_rgb['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x5_depth = output_depth['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_depth = output_depth['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_depth = output_depth['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_depth = output_depth['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_depth = output_depth['x1']  # size=(N, 64, x.H/2,  x.W/2)        

        # Fusion Block : NON-LINEAR WEIGHTED COMBINATION 
        x5 = self.relu(self.fuseBlock_1(x5_rgb, x5_depth, 'x5'))
        x4 = self.relu(self.fuseBlock_1(x4_rgb, x4_depth, 'x4'))
        x3 = self.relu(self.fuseBlock_1(x3_rgb, x3_depth, 'x3'))
        x2 = self.relu(self.fuseBlock_1(x2_rgb, x2_depth, 'x2'))
        x1 = self.relu(self.fuseBlock_1(x1_rgb, x1_depth, 'x1'))

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.dropout(score)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.dropout(score)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.dropout(score)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.dropout(score)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def fuseBlock_1(self, rgb, depth, layer):
        '''
        Fusion Block 1: 
        Weighted mean of RGB and Depth using 1x1 convolution 
        '''
        fused = torch.cat((rgb, depth), dim=1)
        # weight = F.softmax(self.fusionConv[layer].weight, dim=1)
        x = self.fusionConv[layer](fused)
        # x = F.conv2d(fused, weight)

        return x


class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class, p):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class*2, kernel_size=1)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.dropout(score)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.dropout(score)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.dropout(score)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.dropout(score)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class*2, x.H/1, x.W/1)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNDNC(pretrained_net=vgg_model, n_class=3)
