from ssl import get_default_verify_paths
import pyrealsense2 as rs        
import time
import json
import atexit
import cv2
import numpy as np
import itertools


def get_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]

class Camera(rs.pipeline, rs.rs400_advanced_mode):
    '''
    Multifunctional camera wrapper class for D435i sensor
    Stream data and params in real-time.
    Future:
    Custom Filters and post-processing methods
    '''
    def __init__(self, preset = None, intrinsics = None, params = None) -> None:
        '''
        Multifunctional camera wrapper class for D435i sensor
        Stream data and params in real-time. 
        Future:
        Custom Filters
        
        Initialize Camera with inherited rs.pipeline and rs.config base classes
        Example: 


        # Additional params
        intrinsics: Intrinsic matrix
        quality params: User-defined parameter (dict) for depth quality metrics

        '''
        rs.pipeline.__init__(self)
        # rs.config.__init__(self)
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        assert len(list(devices)) == 1, "Either no camera or more than one camera has been connected."
        self.device = devices[0] # This gives back the sensor options
        self.depth_sensor = self.device.first_depth_sensor()
        self.color_sensor = self.device.first_color_sensor()
        self.motion_sensor = self.device.first_motion_sensor()
        rs.rs400_advanced_mode.__init__(self, devices[0])
        # Parameter attribute for depth sensor
        self.param_dict = self.get_depth_sensor_params()
    def start_camera(self, height, width, framerate, streams = None):
        '''
        Start camera streams specified by list streams:
        Arguments: 
        height: int
        width: int
        framerate: int, Default: 30
        streams: List, Default: None
        Optional - Enables motion, color, infrared and depth streams by default for the D435i
        (To be expanded for Lidar and T265 Camera)
        
        
        Returns: None
        '''
        # if streams == None:
        config = rs.config()
        # pipeline = rs.pipeline()
        # config.enable_stream(stream_type = rs.stream.depth, stream_index = -1, format = rs.format.z16, height = height, width = width, framerate = 0)
        # config.enable_stream(stream_type = rs.stream.color, stream_index = -1, format = rs.format.bgr8, height = height, width = width, framerate = 0)
        # # self.enable_stream(rs.stream.infrared, height, width, framerate)
        # config.enable_stream(rs.stream.accel)
        # config.enable_stream(rs.stream.gyro)
        config.enable_stream(rs.stream.depth, width = width, height = height, format = rs.format.z16, framerate = 30)
        config.enable_stream(rs.stream.color, width = width, height = height, format = rs.format.bgr8, framerate = 30)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.start(config)
    def stop_camera(self):
        self.stop()

    def query_sensor_param_bounds(self, params = None, arrayFmt = True):
        '''
        Input: params (Optional): List of requested parameters
        arrayFmt (Default : True): Bool, if return value should be dict or array
        Returns: Upper and Lower bounds of sensor parameters
        '''
        # Basic options
        exposure_min, exposure_max = self.device.query_sensors()[0].get_option_range(rs.option.exposure).min, self.device.query_sensors()[0].get_option_range(rs.option.exposure).max
        gain_min, gain_max = self.device.query_sensors()[0].get_option_range(rs.option.gain).min, self.device.query_sensors()[0].get_option_range(rs.option.gain).max
        laser_power_min, laser_power_max = self.device.query_sensors()[0].get_option_range(rs.option.laser_power).min, self.device.query_sensors()[0].get_option_range(rs.option.laser_power).max
        # Advanced params
        depth_control_min, depth_control_max = self.get_depth_control(1), self.get_depth_control(2)
        depth_control_attrs = [d for d in dir(depth_control_min) if not d.startswith('__')]
        depth_control_min, depth_control_max = [depth_control_min.__getattribute__(d) for d in depth_control_attrs], [depth_control_max.__getattribute__(d) for d in depth_control_attrs]
        min = [[exposure_min], [gain_min], [laser_power_min]]
        min = list(itertools.chain.from_iterable(min))
        max = [[exposure_max], [gain_max], [laser_power_max]]
        max = list(itertools.chain.from_iterable(max))
        return np.asarray(min), np.asarray(max)
    def get_depth_sensor_params(self, params = None, arrayFmt = True):
        '''
        Streams all relevant parameters. [Expandable]
        Current parameter list:
        1. Depth Units : self.device.first_depth_sensor.get_option(rs.option.depth_units)
        2. 
        '''
        # Depth sensor options - Basic
        depth_units = self.depth_sensor.get_option(rs.option.depth_units)
        exposure = self.depth_sensor.get_option(rs.option.exposure)
        gain = self.depth_sensor.get_option(rs.option.gain)
        laser_power = self.depth_sensor.get_option(rs.option.laser_power)
        # Depth sensor - advanced 
        depth_control = self.get_depth_control() # object depth_control of class rs.STDepthControlGroup. Access attributes separately?
        depth_control_attrs = [d for d in dir(depth_control) if not d.startswith('__')]
        depth_control_vals = [depth_control.__getattribute__(d) for d in depth_control_attrs]
        
        depth_table = self.get_depth_table() 
        depth_table_attrs = [d for d in dir(depth_table) if not d.startswith('__')]
        depth_table_vals = [depth_table.__getattribute__(d) for d in depth_table_attrs]
        if arrayFmt == False:
            # emitter_enabled = self.depth_sensor.get_option(rs.option.emitter_enabled)
            basic_options_dict = dict({'depthUnits': depth_units, 'exposure': exposure, 'gain': gain, 'laserPower': laser_power})
            depth_control_dict = {'STDepthControlGroup': {depth_control_attrs[i]: depth_control_vals[i] for i in range(len(depth_control_attrs))}}
            depth_table_dict = {'STDepthTableControl': {depth_table_attrs[i]: depth_table_vals[i] for i in range(len(depth_table_attrs))}}
            # Param dict
            param_dict = basic_options_dict
            param_dict.update(basic_options_dict)
            param_dict.update(depth_control_dict)
            param_dict.update(depth_table_dict)
            return param_dict
        else:
            params = [[exposure], [gain], [laser_power]]
            params = list(itertools.chain.from_iterable(params))
            # params = exposure
            return np.asarray(params)
    def set_depth_sensor_params(self, params, set_and_get = True):
        '''
        Update depth sensor with new params
        Order: Exposure, Gain, Laser_Power, depth_control - Need to
        implement verification for this order somehow...
        Input: params: Array of updated params in exact order
        Return: None or params depending on get
        '''
        STdepthControlGroup = rs.STDepthControlGroup()
        basic_params = params[0], params[1], params[2]
        # depthControlParams = params[3:13]
        # Basic param update
        self.depth_sensor.set_option(rs.option.exposure, params[0])
        time.sleep(1)
        self.depth_sensor.set_option(rs.option.gain, basic_params[1])
        time.sleep(1)
        self.depth_sensor.set_option(rs.option.laser_power, basic_params[2])
        time.sleep(1)
        # # Advanced param update
        # for i, attr in enumerate(get_attributes(STdepthControlGroup)):
        #     setattr(STdepthControlGroup, attr, int(depthControlParams[i]))
        # self.set_depth_control(STdepthControlGroup)
        if set_and_get:
            return self.get_depth_sensor_params()
    def load_preset(self, file):
        jsonObj = json.load(open(file))
        json_string = str(jsonObj).replace("'", '\"')
        self.load_json(json_string)
    def save_json_file(self, params = None):
        print("NotImplemented.")
    # Streaming functions with opencv
    # Helpers
    def __get_camera_info__(self, arg):
        return rs.camera_info(arg)
