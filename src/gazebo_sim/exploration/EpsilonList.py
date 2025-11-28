import argparse
import numpy as np
from . import ExplorationController ;

class EpsListGreedy (ExplorationController):
    def __init__(self):
        self.config, _ = self.parse_args()
        self.max_steps = self.config.max_steps_per_episode +1
        self.epsilon_init = self.config.initial_epsilon
        self.epsilon = [self.epsilon_init]*self.max_steps
        self.epsilon_delta = self.config.epsilon_delta
        self.eps_replay_factor = self.config.eps_replay_factor
        self.eps_min = self.config.final_epsilon
        self.last_dice = -1
        self.start_task = self.config.exploration_start_task

        self.current_step_in_episode = 0

        self.exploration = True ;
    
    def reset_epsilon(self,factor=1.0):
        self.epsilon = [self.epsilon_init * factor]*self.max_steps
    
    def configure(self,curr_task,task):
        if task == self.config.exploration_start_task:
            self.reset_epsilon() # reset epsilon to init
            self.epsilon_delta = self.config.epsilon_delta
        elif curr_task > self.config.exploration_start_task: 
            self.reset_epsilon(self.eps_replay_factor)  # NOTE: scale initial_eps & eps_delta for replay training.        
            self.epsilon_delta = self.epsilon_delta * self.eps_replay_factor 
        elif task < self.config.exploration_start_task: 
              print("----------------------BABBLING_, pure exploration!!") ;
              self.epsilon = [1.0]*self.max_steps ;
              self.epsilon_delta = 0.0 ;
              self.eps_min = 1.0 ;
    
    def query(self):
        self.last_dice = np.random.random()
        return self.last_dice < self.epsilon[self.current_step_in_episode] and self.exploration

    def update_epsilon(self):
        old_eps = self.epsilon[self.current_step_in_episode]
        new_eps = old_eps - self.epsilon_delta if old_eps > self.eps_min else self.eps_min
        print("Update EPS from",old_eps,"to",new_eps,"@",self.current_step_in_episode)
        self.epsilon[self.current_step_in_episode] = new_eps

    def enable_exploration(self):
        self.exploration = True

    def disable_exploration(self):
        self.exploration = False

    def get_current_status(self):
        return (self.epsilon,)

    def define_base_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--seed',     type=int, default=42,                          help='The random seed for the experiment run.')
        
        parser.add_argument('--debug',                   type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        parser.add_argument('--exploration', type=str)
        parser.add_argument('--exploration_start_task',                  type=int, default=0        ,          help='all taskl before this one are babbling')
        parser.add_argument('--max_steps_per_episode',    type=int,                   help='The max number of eps per episode for the epsilon list')
        parser.add_argument('--initial_epsilon',          type=float, default=1.0,    help='The initial probability of choosing a random action.')
        parser.add_argument('--final_epsilon',            type=float, default=0.01,   help='The lowest probability of choosing a random action.')
        parser.add_argument('--epsilon_delta',            type=float, default=0.001,  help='Epsilon reduction factor (stepwise).')
        parser.add_argument('--eps_replay_factor',        type=float, default=0.5,    help='eps start for tasks > 0.')

    def parse_args(self):
        parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.', exit_on_error=False)
        self.define_base_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;
