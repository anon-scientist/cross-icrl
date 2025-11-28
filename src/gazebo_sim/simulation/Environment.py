"""
Abstract Environment Class 
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


from gz.msgs10.laserscan_pb2 import LaserScan ;

from gazebo_sim.simulation.ThreePiManager import ThreePiManager, TwistAction ;

from cl_experiment.parsing import Kwarg_Parser ;

#import gazebo_sim.utils.logger as logger
from cl_experiment.utils import log, change_loglevel ;



class GenericEnvironment(ABC):
    def __init__(self,env_config:dict,step_duration_nsec=100*1000*1000, **kwargs)->None:
        assert env_config != None, "The EnvironmentWrapper needs environment configurations!"
        self.config = self.parse_args(**kwargs)
        env_config["debug"] = self.config.debug ;
        self.step_duration_nsec = step_duration_nsec

        self.training_duration = self.config.training_duration
        self.evaluation_duration = self.config.evaluation_duration
        self.max_steps_per_episode = self.config.max_steps_per_episode
        #self.task_list = self.config.task_list

        self.observation_shape = env_config["observation_shape"]
        task_list = self.parse_task_list(self.config.task_list, len(env_config["tasks"])) ;
        print("Explicit tasks", task_list) ;
        self.tasks = [env_config["tasks"][i] for i in task_list] ;
        self.action_entries = env_config["actions"]
        self.nr_actions = len(self.action_entries)

        self.step_count = 0
        self.task_index = 0
        self.set_manager(env_config) ;

    # overwrite!
    @abstractmethod
    def set_manager(self,env_config):
        pass ;

    # convert expresssions like *,1-15,16 or 31-34 into lists of tasks
    def parse_task_list(self, task_list, nr_tasks):
      print("tasklist", task_list, "nrtasks", nr_tasks) ;
      if task_list.find("*") != -1: return list(range(nr_tasks)) ;
      l = [] ;
      for chunk in task_list.strip().split(","):
        if chunk.find("-") != -1:
          from_task = int(chunk.split("-")[0]) ;
          to_task = int(chunk.split("-")[1]) ;
          l.extend(list(range(from_task, to_task+1))) ;
        else:
          l.append(int(chunk)) ;
      return l ;

    def get_nr_of_tasks(self):
        return len(self.tasks)

    def get_input_dims(self):
        return self.observation_shape

    @abstractmethod
    def get_current_status(self):
        return (0.,)

    # internal method
    def perform_action(self, action_index:int)->None:
        """ high level action execution """
        if self.config.debug=="yes": log.debug(f'action request at tick [{self.step_count}]')

        action = self.action_entries[action_index] # select action to publish to GZ

        if self.config.debug=="yes": 
            log.debug(f'action i={action_index} ({action.wheel_speeds[0]:2.2f}/{action.wheel_speeds[1]:2.2f}) published at tick [{self.step_count}]')

        self.manager.gz_perform_action(action)
        
        
    # performs a task switch. Does not necessarily involve moving the robot, just a change in internal states
    @abstractmethod
    def switch(self,task_index:int)->None:
        pass

    # start of an episode. Returns the first observation without perfoming an action, plus an info dict
    # observations can in general be dicts if numpy arrays (multimodal setup) or simple numpy arrays
    @abstractmethod
    def reset(self)->tuple[dict, dict]:
        self.step_count = 0

    # episode step: returns
    @abstractmethod
    def step(self, action_index:int)->tuple[dict,float,bool,bool,dict]:
        pass

    # internal method
    def compute_reward(self, *args, **kwargs):
        pass

    def parse_args(self, **kwargs):
        parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.', exit_on_error=False)
        parser = Kwarg_Parser(**kwargs) ;

        # ------------------------------------ LEARNER
        parser.add_argument('--debug', type=str,default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        parser.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated result.') ;
        parser.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')
        #parser.add_argument('--task_list', type=str, required=True, nargs="*", help='tasks to execute')
        parser.add_argument('--observations_as_dict', type=str, default='no',  choices=['yes', 'no'], help='what do reset and step return?') ;
        parser.add_argument('--return_modality', type=str, default='vis',  choices=['vis', 'laser','pose'], help='if prev. param is no: what modality should be returned?') ;

        parser.add_argument('--training_duration',        type=int, default=200, help='Defines the number of iterations á training_duration_unit.')
        parser.add_argument('--evaluation_duration',      type=int, default=5,    help='Defines the number of iterations á evaluation_duration_unit.')
        parser.add_argument('--training_duration_unit',   type=str, default='episodes', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) t') ;
        parser.add_argument('--evaluation_duration_unit', type=str, default='episodes',  choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level)') ;
        parser.add_argument('--max_steps_per_episode',    type=int, default=40, help='Sets the number of steps after which an episode gets terminated.')
        parser.add_argument('--start_task',               type=int, default=0        ,          help='')
        parser.add_argument("--task_list", type=str, default="*", nargs="*", required=False, help='tasks to execute') ;

        cfg,unparsed = parser.parse_known_args() ; 
        # wtf??
        self.task_list = cfg.task_list ;
        return cfg ;

    def close(self):
        self.manager.destroy_node()


