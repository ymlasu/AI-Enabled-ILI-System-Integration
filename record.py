'''
Data recording :
This program enables synchronized data storage in the following format.
We also use an extremum seeking controller (ESC) to perform online 
optimization of the depth map. 
Note on ESC:
In the current implementation, ESC does not always run, but only
when the quality metric (loss) increases a specific threshold above the 
local optimum. 
=====
DATA FMT:
RGB-X-Y-Z-ODOM
RGB - Color map from D435i
X, Y, Z - X, Y, Z maps from D435i
ODOM - RTAB-Map odometry output:
Odom format: 3x3 matrix : [gyro 1x3, accel 1x3, stepper_odom 1x3 ]
stepper_odom : [0 0 angle]
=====
'''
import numpy as np
import pyrealsense2 as rs
import cv2
import time
import urllib.request
import argparse
import os


import matplotlib.pyplot as plt
# ROS


from Camera import *
from data_utils import *

# Need to insert required packages to interface with RTAB-Map

np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument("--ip", help = 'Raspberry Pi Addr', default='192.168.120.41')
args = parser.parse_args()

ip = args.ip
link = 'http://' + ip + ':8080'
use_stepper = False



def save_data(frame_id, save_path, rgb, x, y, z, odom_raw = None):
    if frame_id == 10:
        print(rgb.shape, x.shape)
    camera_data = np.dstack((rgb, x,y,z))
    # subplots - temp!
    # cv2.imwrite(save_path + 'rgb_' + str(frame_id) + '.png', rgb)
    # cv2.imwrite(save_path + 'depth_' + str(frame_id) + '.png', z)
    np.save(file = save_path + 'imgs_' + str(frame_id) + '.npy', arr = camera_data)
    if odom_raw.shape:
        np.save(file = save_path + 'odom_' + str(frame_id) + '.npy', arr = odom_raw)

def get_frame(camera):
    frames = camera.wait_for_frames()
    aligned_frame = align.process(frames)
    aligned_color_frame = aligned_frame.get_color_frame()
    aligned_depth_frame = aligned_frame.get_depth_frame()
    return aligned_frame, aligned_color_frame, aligned_depth_frame



def compute_fill_factor(imgs):
    '''
    Given K frames, compute time-averaged fill factor fitness.
    Input: imgs: KxWxH array : either RoI or entire Image
    Returns: fill_factor_cost
    '''
    # For one frame
    num_filled = np.asarray([np.count_nonzero(img) for img in imgs])
    fill_factor = np.asarray([(nf/(imgs.shape[1]*imgs.shape[2])) for nf in num_filled])
    fill_factor_cost = np.mean(np.asarray([-np.log(f) if f != 0 else 100.0 for f in fill_factor]))
    return fill_factor_cost

def ES_step(p_n,i,cES_now,amplitude):
    # ES step for each parameter
    p_next = np.zeros(nES)
    
    # Loop through each parameter
    for j in np.arange(nES):
        p_next[j] = p_n[j] + amplitude*dtES*np.cos(dtES*i*wES[j]+kES*cES_now)*(aES[j]*wES[j])**0.5
    
        # For each new ES value, check that we stay within min/max constraints
        if p_next[j] < -1.0:
            p_next[j] = -1.0
        if p_next[j] > 1.0:
            p_next[j] = 1.0
            
    # Return the next value
    return p_next

# Function that normalizes paramters
def p_normalize(p):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm

# Function that un-normalizes parameters
def p_un_normalize(p):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm


# save
save_path = 'data/records/060622/'
# camera data stream
camera = Camera()
width = 848
height = 480
camera.start_camera(height = height, width = width, framerate = 30)
# camera.load_preset('HighAccuracyPreset.json')

# size = (width, height)
# video = cv2.VideoCapture(0)
# if (video.isOpened() == False): 
#     print("Error reading video file")
# result = cv2.VideoWriter('filename.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)


# ESC params init
eps = 1e-2
p_min, p_max = camera.query_sensor_param_bounds()
nES = len(p_min)
p_diff = p_max - p_min
p_ave = (p_max + p_min)/2.0
# pES = np.zeros([ES_steps,nES])
wES = np.linspace(1.0,1.75,nES)
dtES = 2*np.pi/(10*np.max(wES))
oscillation_size = 0.1
aES = wES*(oscillation_size)**2
kES = 0.1
decay_rate = 0.99
amplitude = 1.0

align = rs.align(rs.stream.color)
pc = rs.pointcloud()
frame_id = 0
try:
    while True:
        start_time = time.time()
        if use_stepper == True:
            stepper_odom = get_stepper_odom(link)
        amplitude = 1.0
        frame_id = frame_id + 1
        # print(frame_id)
        start_time = time.time()
        frame, color_frame, depth_frame= get_frame(camera)
        accel = get_accel_data(frame)
        gyro = get_gyro_data(frame)
        if use_stepper == True:
            odom_raw = np.asarray([gyro, accel, stepper_odom])
        else:
            odom_raw = np.asarray([gyro, accel])
        color = np.asarray(color_frame.get_data())
        depth = np.asarray(depth_frame.get_data())
        depth = np.expand_dims(depth, axis = 0)
        if frame_id == 1:
            prev_cost = 1e10
        cost = compute_fill_factor(depth)
        j = 0
        while (cost > 1.0 and frame_id < 100):
            j = j + 1
            print("ESC Iteration number: ", j)
            print("Hold Camera still ... ESC working.")
            frame_id = frame_id + 1
            pES = camera.get_depth_sensor_params()
            pES_n = p_normalize(pES)
            pES_n = ES_step(pES_n,j,cost,amplitude)
            pES = p_un_normalize(pES_n)
            prev_cost = cost
            camera.set_depth_sensor_params(pES)
            print("Params modified for iteration : ", j)
            frame, color_frame, depth_frame = get_frame(camera)
            depth = np.asarray(depth_frame.get_data())
            color = np.asarray(color_frame.get_data())
            depth = np.expand_dims(depth, axis = 0)
            cost = compute_fill_factor(depth)
            print("Cost: ", cost)
            print("Improvement: ", np.abs(cost - prev_cost))
            amplitude = amplitude*decay_rate
            if j > 100 and cost < 0.25:
                print("Camera param calibration complete.")
                break
        pointsContainer = pc.calculate(depth_frame)
        points = np.asarray(pointsContainer.get_vertices())
        points = points.view(np.float32).reshape(points.shape + (-1,))
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        x = x.reshape((height,width, 1))
        y = y.reshape((height,width, 1))
        z = z.reshape((height,width, 1))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth[0], alpha=0.03), cv2.COLORMAP_JET)
        # print(color.shape, depth_colormap.shape)
        # if frame_id % 1 == 0:
        #     save_data(frame_id=frame_id, save_path=save_path, rgb = color, x = x, y = y, z = z
        #     , odom_raw = odom_raw)
        
        cv2.imshow("Frame",np.hstack([color, depth_colormap]))   #show captured frame
        #press 'q' to break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- Data from RTAB Map --- #
        '''
        Notes:
        Insert relevant calls to data outs from rtabmap-ros for odom data
        '''

        # Call save_data to write data into file (continuous? too much space(!!))


        print("--- %s Hz ---" % (1/(time.time() - start_time)))
finally:
    # video.release()
    # result.release()
    cv2.destroyAllWindows()
    # print("The video was successfully saved")
    print("stopping the camera")
    camera.stop()
