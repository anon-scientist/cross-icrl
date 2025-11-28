import argparse
import numpy as np
from gazebo_sim.exploration.BaseExploration import BaseExploration
import gazebo_sim.utils.logger as logger

from . import ExplorationController ;

def log(*values:object, sep=' ', end='\n', file = "EpsilonProtos.txt", level = logger.LogLevel.INFO):
  logger.log(*values,sep=sep,end=end,file=file,level=level)

class EpsProtosGreedy(ExplorationController):
    def __init__(self, QGMM_K):
        super().__init__()
        self.config, _ = self.parse_args()
        self.QGMM_K = QGMM_K
        self.epsilon_init = self.config.initial_epsilon
        self.epsilon = [self.epsilon_init]*self.QGMM_K
        self.epsilon_delta = self.config.epsilon_delta
        self.eps_replay_factor = self.config.eps_replay_factor
        self.eps_min = self.config.final_epsilon
        self.last_dice = -1

        self.proto_threshold = self.config.exp_proto_threshold

        self.current_proto = 0
        self.proto_certainty = 0.0
 
        self.babbeling = True
    
    def reset(self,factor=1.0):
        self.epsilon = [self.epsilon_init*factor]*self.QGMM_K
    
    def configure(self,curr_task,task):
        if task == self.start_task:
            self.babbeling = False
            self.reset() # reset epsilon to init
            self.epsilon_delta = self.config.epsilon_delta
        elif curr_task >= self.start_task:
            self.babbeling = False
        elif task < self.start_task: 
              log("----------------------BABBLING, pure exploration!!") ;
              self.babbeling = True
    
    def query(self):
        self.last_dice = np.random.random()
        if self.proto_certainty <= self.proto_threshold:
            return self.exploration
        return self.last_dice < self.epsilon[self.current_proto] and self.exploration
    
    def update(self):
        if self.babbeling:
            log("Babbeling! No eps change!")
            return
        if self.proto_certainty <= self.proto_threshold:
            log("---proto certainty is under threshold!!!")
        else:
            old_eps = self.epsilon[self.current_proto]
            new_eps = old_eps - self.epsilon_delta if old_eps > self.eps_min else self.eps_min
            print("Update EPS from",old_eps,"to",new_eps,"@",self.current_proto)
            self.epsilon[self.current_proto] = new_eps

    def enable_exploration(self):
        self.exploration = True

    def disable_exploration(self):
        self.exploration = False

    def get_current_status(self):
        return ("NOSTATUS",)

    def define_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--initial_epsilon',          type=float, default=1.0,    help='The initial probability of choosing a random action.')
        parser.add_argument('--final_epsilon',            type=float, default=0.01,   help='The lowest probability of choosing a random action.')
        parser.add_argument('--epsilon_delta',            type=float, default=0.001,  help='Epsilon reduction factor (stepwise).')
        parser.add_argument('--eps_replay_factor',        type=float, default=0.5,    help='eps start for tasks > 0.')
        parser.add_argument('--exp_proto_threshold',      type=float, default=0.9,    help='The threshold when an observation is attributed to a prototype. If the models certainty is above this threshold the observation could be mapped to an existing prototype!')
    
    def parse_args(self):
        parser = argparse.ArgumentParser('EpsProtos', 'argparser for EpsProtos.', exit_on_error=False)
        self.define_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;
