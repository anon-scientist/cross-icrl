import time ;
import math ;
import argparse ;
import numpy as np ;

from abc import ABC, abstractmethod
from typing import * ;
from threading import Lock ;
from scipy.spatial.transform import Rotation

from gz.transport13 import Node ;
from gz.msgs10.empty_pb2 import Empty ;
from gz.msgs10.scene_pb2 import Scene ;
from gz.msgs10.image_pb2 import Image ;
from gz.msgs10.twist_pb2 import Twist ;
from gz.msgs10.world_control_pb2 import WorldControl ;
from gz.msgs10.boolean_pb2 import Boolean ;
from gz.msgs10.pose_pb2 import Pose ;
from gz.msgs10.pose_v_pb2 import Pose_V ;
from gz.msgs10.clock_pb2 import Clock ;


from gz.msgs10.laserscan_pb2 import LaserScan ;

from gazebo_sim.simulation.ThreePiManager import ThreePiManager, TwistAction ;

from cl_experiment.parsing import Kwarg_Parser ;

#import gazebo_sim.utils.logger as logger
from cl_experiment.utils import log, change_loglevel ;

from .Environment import GenericEnvironment ;




# simple class for describing a starting point and an orientation (can be > 1 per task) on the ground plane
class ThreePiTask(object):
    class Transform(object):
        def __init__(self,position:Tuple[float,float,float],euler_rotation:Tuple[int,int,int]):
            self.position = position
            orientation = Rotation.from_euler('xyz',euler_rotation,degrees=True).as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;
        def add_rotation(self, euler_rotation:Tuple[int,int,int]):
            rot_modifier = Rotation.from_euler('xyz',euler_rotation,degrees=True)
            current_orientation = Rotation.from_quat(self.orientation)
            orientation = current_orientation * rot_modifier
            orientation = orientation.as_quat(canonical=False)
            self.orientation = [float(o) for o in orientation] ;

    def __init__(self, name:str, start_points:dict=None, **kwargs) -> None:
        self.name = name
        self.starting_points = start_points if start_points is not None else dict() ;
        self.settings = kwargs

    def add_starting_point(self, entry):
        self.starting_points[entry["name"]] = entry ;

    def get_random_start(self)->dict:
        indices = list(self.starting_points.keys()) ;
        random_index = np.random.choice(indices)
        print("@@@@@@Task ", self.name, "choosing starting point", self.starting_points[random_index]) ;
        return self.starting_points[random_index] ;

    def get(self,key:str):
        return self.settings.get(key,None)

    def get_name(self):
      return self.name ;



class ThreePiEnvironment(GenericEnvironment):
    def __init__(self,env_config:dict,step_duration_nsec=100*1000*1000, **kwargs)->None:
      GenericEnvironment.__init__(self, env_config, step_duration_nsec, **kwargs) ;

    def set_manager(self,env_config):
        self.manager = ThreePiManager(env_config) ;

    # primary observation modality, need not be vision. Attempts to give a new observation only if 
    # the timstamp in the respective messages has avanced sufficiently to keep up the franerate
    # TODO make this generic so it is not tied to lidar and vision!
    def get_observation(self, nsec=None):
        if nsec is None:
            nsec = self.step_duration_nsec
        t0 = self.manager.get_last_obs_time()
        #log(s)
        i = 0 ;
        last = t0 ;
        while ((self.manager.get_last_obs_time() - t0) < nsec):
            time.sleep(0.001) ;
            if self.manager.get_last_obs_time() != last: 
              last = self.manager.get_last_obs_time() ;
              i += 1 ;
              #log("frame while waiting: i, t0,last,delta,nsec = ", i, t0, last, last-t0, nsec) ;

            pass ;

        #print("frame accepted at", self.manager.get_last_obs_time()-t0) ;

        # once we have a vision frame, wait for the next LIDAR frame to arrive, and give it back together with the others
        # we assume that Lidar has a much higher freuqency than vision, so this will not impact vision frequency 
        # goal is that LIDAR mesaurement should be *newer* than vision rather the other way around
 
        while True:
          lidar_stamp = self.manager.get_last_lidar_obs_time() ;
          #print("lidar", lidar_stamp, "vision", self.manager.get_last_obs_time()) ;
          if lidar_stamp >= self.manager.get_last_obs_time():
            break ;
          time.sleep(0.001);
        #print("return");
        return self.manager.get_data(), self.manager.get_last_range(), self.manager.get_position(), self.manager.get_last_contact_obs() ;

    # this always returns the last lidar observation, does not wait. We assume that waiting for primary modality is enough.
    def get_range(self, nsec=None):
        response = self.manager.get_last_range()
        return response ;


