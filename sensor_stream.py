#!/home/rrathnak/anaconda3/envs/rgbd/bin/python
'''
Spawns an instance of the camera class for streaming data from the camera
'''
import pyrealsense2 as rs
from Camera import *
from data_utils import *

import cv2
import argparse


np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument("--ip", help = 'Raspberry Pi Addr', default='192.168.120.41')
args = parser.parse_args()

ip = args.ip
link = 'http://' + ip + ':8080'
use_stepper = False



def get_frame(camera):
    frames = camera.wait_for_frames()
    aligned_frame = align.process(frames)
    aligned_color_frame = aligned_frame.get_color_frame()
    aligned_depth_frame = aligned_frame.get_depth_frame()
    return aligned_frame, aligned_color_frame, aligned_depth_frame


camera = Camera()
width = 848
height = 480
camera.start_camera(height = height, width = width, framerate = 30)
camera.load_preset('HighAccuracyPreset.json')
align = rs.align(rs.stream.color)
pc = rs.pointcloud()
try:
    while True:
        start_time = time.time()
        if use_stepper == True:
            stepper_odom = get_stepper_odom(link)
        amplitude = 1.0
        # print(frame_id)
        start_time = time.time()
        frame, color_frame, depth_frame= get_frame(camera)
        accel = get_accel_data(frame)
        gyro = get_gyro_data(frame)
        if use_stepper == True:
            odom_raw = np.asarray([gyro, accel, stepper_odom])
        else:
            odom_raw = np.asarray([gyro, accel])
        print(odom_raw)
        print("--- %s Hz ---" % (1/(time.time() - start_time)))
finally:
    # video.release()
    # result.release()
    # cv2.destroyAllWindows()
    # print("The video was successfully saved")
    print("stopping the camera")
    camera.stop()
    