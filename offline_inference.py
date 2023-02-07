'''
inference.py
Functionality:
The goal of this program is to accept data from the Camera stream, load in a trained ML model, and produce predictions,
uncertainties, measurements and risk assessments. 
**** This is the OFFLINE INFERENCE CODE for hardware-software integration ****
Expected IO: 
Input: Data in the form of images
Output: Corresponding results, convert to GIF for demo.
===============================================
'''

# Core
import numpy as np
import cv2
import time
import sys
import os
import copy
import GPUtil
import shutil
import csv
import time 
import argparse
import logging
import json


# Data processing
import csv
import pandas as pd

# Torch
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms     
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Visualization packages
from visdom import Visdom
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
import imshowpair

# Modules
from network import *
from dataset import *
from risk_assessment import measure

# Utilities
from image_utils import *
from data_utils import *
from network_utils import *
from metrics import *

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help = 'OPTIONS: Pipeline, RoadCracks', default = 'Pipeline')
parser.add_argument("--data_fmt", help = 'OPTIONS: rgb, rgb-d, rgb-dc, rgb-dnc', default = 'rgb')
parser.add_argument("--model_path", help = 'full path to inference model weights', 
default= 'models/ASU_FULL_PredictiveVariance_0.1dropout_39Train_ExtraData_RGB_0/best/best_model.pt')
parser.add_argument("--root_dir", help = "Root directory for images", default = 'data/records/060422')
parser.add_argument("--mc_samples", help = 'Number of MC Dropout samples (int) ', default = '10')
args = parser.parse_args()

# Setup
dataset = args.dataset
data_fmt = args.data_fmt
model_path = args.model_path
mc_samples = args.mc_samples
root_dir = args.root_dir
labels_avail = False
p = 0.1 # Dropout ratio
batch_size = 1 # max this out!
dataset_options = ['Pipeline, RoadCracks']
if dataset == 'Pipeline':
    num_classes = 3
else:
    num_classes = 2

'''
Results directory
'''
results_dir = 'results/run0_640x480/'

'''
Data FMT:
RGB-X-Y-Z-Odom
This data is stored offline using record.py during inspection run.
- Data processing w/ numpy to obtain RGB-DNC from RGB-XYZ
- PyTorch dataloader 
- Proceed as usual for inference
- Plug in measurement module (improvements needed!)
- Risk assessment from measurement module (improvements needed!)
'''

# Load model 
vgg_model = VGGNet()
if data_fmt == 'rgbd':
    input_modalities = ['rgb', 'depth']
    net = FCNDepth(pretrained_net = vgg_model, n_class = num_classes, p = p)
elif data_fmt == 'rgbdnc':
    input_modalities = ['rgb', 'depth', 'normal', 'curvature']
    net = FCNDNC(pretrained_net = vgg_model, n_class = num_classes, p = p)
elif data_fmt == 'rgbdc':
    input_modalities = ['rgb', 'depth', 'curvature']
    net = FCNDC(pretrained_net = vgg_model, n_class = num_classes, p = p)
else:
    input_modalities = ['rgb']
    net = FCNs(pretrained_net = vgg_model, n_class = num_classes, p = p)
net, epoch = load_ckp(model_path, net)
vgg_model = vgg_model.to(device)
net = net.to(device)
net.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


print('Preparing dataset...')
print('Creating directories...')
# prepare_images_for_offline_inference(root_dir=root_dir, modalities = ['rgb', 'depth'])

# prepare_npy_for_offline_inference(root_dir = root_dir, modalities = ['rgb', 'depth', 'x', 'y', 'z'])
print('Creating dataloaders...')
val_dataset = DefectDataset(root_dir = root_dir, num_classes = num_classes, input_modalities = input_modalities,
image_set = 'val')
val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

print("Validating at epoch: {:.4f}".format(epoch))
measurements = []
with torch.no_grad():
    net.dropout.train()
    softmax = nn.Softmax(dim = 1)
    for iter, (img_name, data, data_for_measurement) in enumerate(val_dataloader):
        start_time = time.time()
        sampled_outs = []
        outs_sm = []
        for modality in input_modalities:
            data[modality] = data[modality].to(device)
        if labels_avail:
            label = label.to(device)
        # Predicted aleatoric variance from a single pass
        aleatoric_uncertainty = net(data[modality])[:, num_classes:, :, :]
        assert aleatoric_uncertainty.shape[1] == num_classes, "Aleatoric uncertainty shape error."
        aleatoric_uncertainty = np.exp(aleatoric_uncertainty.detach().clone().cpu().numpy())
        # Sampled epistemic uncertainty
        for i in range(int(mc_samples)):
            sampled_outs.append(net(data[modality]))
        for out in sampled_outs:
            N, _, h, w = out.shape
            out_ = out[:, :num_classes, :, : ]
            out__ = softmax(out_)
            outs_sm.append(out__.cpu().numpy())
        
        mean_output = np.mean(np.stack(outs_sm), axis = 0) # TODO: Threshold this mean output (int?)
        N, _, h, w = mean_output.shape
        pred = mean_output.transpose(0, 2, 3, 1).reshape(-1, num_classes).argmax(axis=1).reshape(N, h, w)
        # cv2.imwrite('results/060422/' + img_name[0] + '.jpg', 
        # cv2.hconcat([(invTrans(data[modality][0].detach().cpu())*255).numpy().astype('uint8')[0], (pred[0]*50).astype('uint8')]))
        # imshowpair.imshowpair(data[modality][0][0].detach().cpu().numpy(), pred[0])
        # plt.savefig('results/060422/' + img_name[0] + '.png')
        # plt.close()
        epistemic_uncertainty = np.mean(np.stack(outs_sm)**2, axis = 0) - (np.mean(np.stack(outs_sm), axis = 0))**2 # Batches x num_classes x W x H
        classwise_epistemic_uncertainty = np.mean(epistemic_uncertainty, axis = (2,3))
        classwise_aleatoric_uncertainty = np.mean(aleatoric_uncertainty, axis = (2,3))
        # TODO: Check data_for_measurement argument input format 
        print("Inference complete. Measuring defects for index ", img_name)
        measurements.append(measure(img_name, pred, aleatoric_uncertainty, data_for_measurement['x'][0][0].detach().numpy(), data_for_measurement['y'][0][0].detach().numpy(), data_for_measurement['z'][0][0].detach().numpy()))
        print("--- %s Hz ---" % (1/(time.time() - start_time)))
# with open('measurements.txt', 'w') as f:
#     json.dump(str(measurements), f)
# f.close()