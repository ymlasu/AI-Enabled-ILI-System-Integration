'''
Helpers for neural networks
'''
import torch

def load_ckp(checkpoint_fpath, net, optimizer = None):
    checkpoint = torch.load(checkpoint_fpath)
    net.load_state_dict(checkpoint['net_state_dict'])
    if optimizer is not None: 
        optimizer.load_state_dict(checkpoint['optimizer'])
        return net, optimizer, checkpoint['epoch']
    return net, checkpoint['epoch']
    
