import os
import glob
import re
import shutil
import urllib.request

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

global_modalities = [['rgb'], ['rgb', 'depth'], ['rgb', 'depth', 'normal'], 
['rgb', 'depth', 'normal', 'curvature']]

def fillMissingValues(target_for_interp, copy=True, 
                      interpolator=LinearNDInterpolator): 
    if copy: 
        target_for_interp = target_for_interp.copy()
    def getPixelsForInterp(img): 
        """
        Calculates a mask of pixels neighboring invalid values - 
           to use for interpolation. 
        """
        # mask invalid pixels
        invalid_mask = np.isnan(img) + (img == 0) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #dilate to mark borders around invalid regions
        dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel, 
                          borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask
    # Mask pixels for interpolation
    mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
    # Interpolate only holes, only using these pixels
    points = np.argwhere(mask_for_interp)
    values = target_for_interp[mask_for_interp]
    interp = interpolator(points, values)
    target_for_interp[invalid_mask] = interp(np.argwhere(invalid_mask))
    return target_for_interp


def npy_to_imgs(npy):
    imgs = np.load(npy)
    rgb = imgs[:,200:,0:3]
    x = imgs[:,200:,3]
    y = imgs[:,200:,4]
    z = imgs[:,200:,5]
    return rgb, x, y, z


def prepare_images_for_offline_inference(root_dir, modalities = None):
    '''
    Folder structure creation for PyTorch dataset class if images are saved directly
    in record.py
    '''
    assert modalities in global_modalities, "Modalities incorrectly defined."
    for modality in modalities:
        dir = root_dir + '/' + modality + '/'
        if not os.path.exists(dir):
            print('creating directory: ', dir)
            os.mkdir(dir)
        files = glob.glob(root_dir + '/' + modality + '_*.png')
        for f in files: 
            new_name = f.replace(root_dir + '/' + modality + '_', '')
            shutil.move(f, dir + new_name)
    files = glob.glob(root_dir + modality + '/' + '*.png')
    f = open(root_dir + 'val.txt', 'w')
    for img in files: f.write(os.path.basename(img) + "\n")
    f.close()

def prepare_npy_for_offline_inference(root_dir, modalities):
    for modality in modalities:
        dir = root_dir + '/' + modality + '/'
        if not os.path.exists(dir):
                    print('creating directory: ', dir)
                    os.mkdir(dir)
    imgs_files = glob.glob(root_dir + '/imgs' + '_*.npy')
    odom_files = glob.glob(root_dir + '/odom' + '_*.npy')
    for file in imgs_files:
        rgb, x, y, z = npy_to_imgs(file)
        # x = cv2.normalize(fillMissingValues(x), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
        # y = cv2.normalize(fillMissingValues(y), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
        # z = cv2.normalize(fillMissingValues(z), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
        x = fillMissingValues(x)
        y = fillMissingValues(y)
        z = fillMissingValues(z)
        save_filename = os.path.splitext(os.path.basename(file).replace(os.path.basename(file)[0:5],''))[0]
        print(save_filename)
        print(root_dir)
        cv2.imwrite(root_dir + '/rgb/' + save_filename + '.png', rgb)
        cv2.imwrite(root_dir + '/depth/' + save_filename + '.png', z)
        
        np.save(root_dir + '/x/' + save_filename + '.npy', x)
        np.save(root_dir + '/y/' + save_filename + '.npy', y)
        np.save(root_dir + '/z/' + save_filename + '.npy', z)
    files = glob.glob(root_dir + '/rgb/' + '*.png')
    f = open(root_dir + '/val.txt', 'w')
    for img in files: f.write(os.path.basename(img) + "\n")
    return 0


def npy_format_visualization(load_dir, save_dir):
    '''
    Helper to visualize npy data
    '''
    files = glob.glob(load_dir + 'imgs_*.npy')
    for file in files:
        imgs = np.load(file)
        rgb = imgs[:,:,0:3]
        x = imgs[:,:,3]
        y = imgs[:,:,4]
        z = imgs[:,:,5]
    return rgb,x,y,z

def get_gyro_data(frame):
    gyro = frame[3].as_motion_frame().get_motion_data()
    return np.asarray([gyro.x, gyro.y, gyro.z])

def get_accel_data(frame):
    accel = frame[2].as_motion_frame().get_motion_data()
    return np.asarray([accel.x, accel.y, accel.z])

def get_stepper_odom(link):
    f = urllib.request.urlopen(link)
    myfile = f.read()
    myfile.split()
    return np.asarray([0,0,float(myfile)])
