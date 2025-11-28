"""
Standalone class for controlling a gz-simulated 2-wheeled differential drive robot via gazebo messages
- controlled by Twist actions, use TwistAction class
- equipped with vision and lidar sensors, fake pose sensor, odometry sensor and contact sensor
- independent of Tasks, knows nothing of trhe environment. It simply checks whether the simulation is on and does nothing else
"""
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
from gz.msgs10.odometry_pb2 import Odometry ;
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.laserscan_pb2 import LaserScan ;
from gz.msgs10.entity_factory_pb2 import EntityFactory ;
from gz.msgs10.entity_pb2 import Entity ;

from cl_experiment.utils import log, change_loglevel ;


class TwistAction():
    """ Basic action for the 3pi two-wheeled robot """
    def __init__(self, name, wheel_speeds, separation=.1):
        self.name = name
        self.wheel_speeds = wheel_speeds

        self.action = Twist()
        if wheel_speeds[0]+wheel_speeds[0] != 0:
            self.action.linear.x = (wheel_speeds[0] + wheel_speeds[1]) / 2
            self.action.angular.z = (wheel_speeds[0] - wheel_speeds[1]) / separation

    def return_instruction(self):
        """ Returns a gz-transport message """
        return self.action

    def to_string(self):
        return f'{self.name}: {self.wheel_speeds}'

    def __str__(self):
      return f"WheelSpeeds={self.wheel_speeds[0]}/{self.wheel_speeds[1]}" ;


class ThreePiManager(Node):

    def __init__(self,env_config):
        super().__init__()

        self.mutex = Lock()

        self.env_config = env_config
        self.env_config["debug"] = self.env_config.get("debug", "no") ;

        self.robot_name = self.env_config["robot_name"]
        self.vehicle_prefix = self.env_config["vehicle_prefix"]
        self.world_name = self.env_config["world_name"]
        self.lidar_topic = self.env_config.get("lidar", None) ;
        self.contact_topic = self.env_config.get("contact_topic", None) ;
        self.odo_topic = self.env_config.get("odometry_topic", None) ;
        self.camera_topic = self.env_config["camera_topic"]

        self.step = 0
        self.last_obs_time = 0
        
        self.last_range = []

        #maybe for later, for debugging...
        #if self.subscribe(Clock,f'/clock',self.gz_handle_observation_callback):
        #  print("subscribed to /clock") ;
        if self.subscribe(Image,f'{self.camera_topic}',self.gz_handle_observation_callback):
            print("subscribed to Camera!")

        if self.subscribe(Pose_V,f'{self.world_name}/dynamic_pose/info',self.gz_handle_dynamic_pose_callback):
            print("Subscribed to dynamic_pose/info!")

        if self.odo_topic is not None:
          if self.subscribe(Odometry,f'{self.odo_topic}',self.gz_handle_odo_callback):
              print("Subscribed to odometry " + self.odo_topic) ;
            
        if self.lidar_topic is not None:
          if self.subscribe(LaserScan,f'{self.lidar_topic}',self.gz_handle_lidar_range_callback):
              print("subscribed to Lidar-Sensor!" + self.lidar_topic)

        if self.contact_topic is not None:
          if self.subscribe(Contacts,f'{self.contact_topic}',self.gz_handle_contact_callback):
              print("subscribed to Contact-Sensor!" + self.contact_topic)

        self.gz_action = self.advertise(f'{self.vehicle_prefix}/motor',Twist)
        self.stop_action = TwistAction("stop",[0.,0.]).return_instruction() ;

        self.wait_for_simulation()

        self.spawn_service_name = f'{self.world_name}/create'
        self.spawn_robot_obj = EntityFactory() ;

        self.remove_service_name = f'{self.world_name}/remove'
        self.remove_robot_obj = Entity() ;

        self.world_control_service = f'{self.world_name}/control'
        self.res_req = WorldControl()
        self.set_pose_service = f'{self.world_name}/set_pose'
        self.pos_req = Pose()

        self.last_range = None ; self.last_contact_obs = None ;
        self.reset_robot_name_in_sim() ;

    def wait_for_simulation(self):
        """ check whether the simulation is on by querying it for all models in the scene """
        response = self.request_scene()
        for m in response.model:
          print("Model in scene", m.name) ;

    def request_scene(self):
        result = False;
        start_time = time.time()
        while result is False:
            print(f'#{self.world_name}/scene/info')
            # Request the scene information
            result, response = self.request(f'{self.world_name}/scene/info', Empty(), Empty, Scene, 1)
            print(f'\rWaiting for simulator... {(time.time() - start_time):.2f} sec', end='')
            time.sleep(0.1)
        print('\nScene received!')
        return response

    def get_step(self):
        return self.step

    # return last vision data
    def get_data(self):
        return self.data

    # return last pose "sensor" data
    def get_position(self):
        return self.position

    # return last pose "sensor" data
    def get_orientation(self):
        return self.orientation
    
    def get_orientation_euler(self):
        return Rotation.from_quat(self.orientation)
        

    def get_last_obs_time(self):
        return self.last_obs_time

    def get_last_lidar_obs_time(self):
        return self.last_lidar_obs_time

    # return last lidar obs
    def get_last_range(self):
        return self.last_range

    # return last contact obs
    def get_last_contact_obs(self):
      return self.last_contact_obs ;
    
    def convert_image_msg(self, msg):
        return np.frombuffer(msg.data,dtype=np.uint8).astype(np.float32).reshape(msg.height,msg.width,3) / 255. ;

    # data from dynamic_pose messages/topic
    def get_robot_name_in_sim(self):
      return self.robot_name_in_sim ;

    def reset_robot_name_in_sim(self):
      self.robot_name_in_sim = None;

    # --------- CALLBACKS -----------------------------

    # dummy for now
    def gz_handle_odo_callback(self, msg):
      pass ;
      #print(msg.twist.linear.x, msg.twist.angular.z) ;

    def reset_contact_state(self):
      self.last_contact_obs = None ;
      self.last_contact_time = -1 ;

    def gz_handle_contact_callback(self, msg):
        with self.mutex:
          self.last_contact_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec ;
          self.last_contact_obs = msg.contact ;
          #print("<<<<<<<<<<<<<<<<CONTACT") ;

    def gz_handle_observation_callback(self,msg):
        with self.mutex:
            #print("OBS") ;
            self.data = msg
            self.last_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec
            #time.sleep(0.001) ;
            #print("OBS", self.last_obs_time) ;

    def gz_handle_dynamic_pose_callback(self,msg):
        with self.mutex:
            self.robot_name_in_sim = None ;
            for p in msg.pose:
                if p.name == self.robot_name:
                    self.position = [p.position.x,p.position.y,p.position.z];
                    self.orientation = [p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]
                    self.robot_name_in_sim = p.name ;
                    #print("Robot name", p.name) ;
                    #print(self.position, 2*math.asin(p.orientation.z) / 3.14 * 180, "deg")
                    return;
            if self.env_config["debug"]=="yes": print(f"THERE WAS NO\033[92m {self.robot_name}\033[0m IN THE SIMULATION!")

    def gz_handle_lidar_range_callback(self,msg):
        with self.mutex:
            #print("lidar", msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec) ;
            if msg.ranges:
                self.last_range = np.array(msg.ranges)
                #good_ranges = self.last_range [np.logical_not(np.isinf(self.last_range))]
                #print("LIDAR", good_ranges.min()) ;
                self.last_lidar_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nsec
            return;

    # ------- central method for executing a given TwistAction ....................
    def gz_perform_action(self, action:TwistAction):
        self.step += 1
        self.gz_action.publish(action.return_instruction())
        if self.env_config["debug"]=="yes": print(f'action published: ', action.to_string())

    def gz_perform_action_stop(self):
        self.gz_action.publish(self.stop_action) ;
        if self.env_config["debug"]=="yes": print(f'stop published!')


    # place robot at a given position and orientation
    def perform_reset(self, position, orientation):
        self.position = position
        self.set_entity_pose_request(self.robot_name,self.position,orientation)

    # call the gazebo service for setting poses
    def set_entity_pose_request(self, name, position, orientation):
        self.pos_req.name = name
        self.pos_req.position.x = position[0]
        self.pos_req.position.y = position[1]
        self.pos_req.position.z = position[2]
        self.pos_req.orientation.x = orientation[0]
        self.pos_req.orientation.y = orientation[1]
        self.pos_req.orientation.z = orientation[2]
        self.pos_req.orientation.w = orientation[3]

        result = False ;
        while result == False:
          result, response = self.request(self.set_pose_service, self.pos_req, Pose, Boolean, 1) ;
          time.sleep(0.01)
          if self.env_config["debug"]=="yes": print(result, response.data)
          if response.data == True: break ;
        """
        # TODO: replace code by this one?
        result, response = self.request(self.set_pose_service, self.pos_req, Pose, Boolean, 1) ;
        if result == False:
          if self.env_config["debug"]=="yes": print("timeout!") ;
        else:
          if self.env_config["debug"]=="yes": print("Success? ", response.data) ;
        """




    def reset_world(self):
        self.res_req.pause = False ;
        self.res_req.reset.all = True  ;
        self.res_req.reset.time_only = False  ;
        self.res_req.reset.model_only = False ;
        if self.env_config["debug"]=="yes": print(f'mdeol reset request !')

        result, response = self.request(self.world_control_service, self.res_req, WorldControl, Boolean, 1) ;
        if result == False:
          if self.env_config["debug"]=="yes": print("timeout!") ;
        else:
          if self.env_config["debug"]=="yes": print("Success? ", response.data) ;



    # pause simulation. Very important!!
    def trigger_pause(self, pause):
        self.res_req.pause = pause
        if self.env_config["debug"]=="yes": print(f'pause={pause} request !')

        result = False ;
        while result == False:
          result, response = self.request(self.world_control_service, self.res_req, WorldControl, Boolean, 1) ;
          time.sleep(0.01)
          if response.data == True: break ;
        """
        # TODO: replace code by this one?
        result, response = self.request(self.set_pose_service, self.pos_req, Pose, Boolean, 1) ;
        if result == False:
          if self.env_config["debug"]=="yes": print("timeout!") ;
        else:
          if self.env_config["debug"]=="yes": print("Success? ", response.data) ;
        """
        if self.env_config["debug"]=="yes": print(f'pause={pause} request done!')


    # spawn a new robot model instance w same name?
    def spawn_robot(self, position, orientation):
        if self.env_config["debug"]=="yes": print(f'spawn request !') ;
        self.spawn_robot_obj.sdf_filename = f"{self.robot_name}/model.sdf" ;
        self.spawn_robot_obj.name = f"{self.robot_name}" ;
        self.spawn_robot_obj.allow_renaming = 1 ;

        self.spawn_robot_obj.pose.position.x = position[0]
        self.spawn_robot_obj.pose.position.y = position[1]
        self.spawn_robot_obj.pose.position.z = position[2]

        self.spawn_robot_obj.pose.orientation.x = orientation[0] ;
        self.spawn_robot_obj.pose.orientation.y = orientation[1] ;
        self.spawn_robot_obj.pose.orientation.z = orientation[2] ;
        self.spawn_robot_obj.pose.orientation.w = orientation[3] ;

	
        result, response = self.request(self.spawn_service_name, self.spawn_robot_obj, EntityFactory, Boolean, 5) ;
        if result == False:
          if self.env_config["debug"]=="yes": print("timeout!") ;
        else:
          if self.env_config["debug"]=="yes": print("Success? ", response.data) ;
        if self.env_config["debug"]=="yes": print(f'spawn request done!')

    # spawn a new robot model instance w same name?
    def remove_robot(self):
        if self.env_config["debug"]=="yes": print(f'remove request for {self.robot_name}!') ;
        self.remove_robot_obj.type = 2 ; # Model
        self.remove_robot_obj.name = self.robot_name ;

        result, response = self.request(self.remove_service_name, self.remove_robot_obj, Entity, Boolean, 5) ;
        if result == False:
          if self.env_config["debug"]=="yes": print("timeout!") ;
        else:
          if self.env_config["debug"]=="yes": print("Success? ", response.data) ;
        
          #if response.data == True: break ;
        if self.env_config["debug"]=="yes": print(f'remove request done!')

