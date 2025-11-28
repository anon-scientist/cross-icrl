import argparse
import numpy as np

from gazebo_sim.exploration import ExplorationController ;


import gazebo_sim.utils.logger as logger

def log(*values:object, sep=' ', end='\n', file = "EpsilonGreedy.txt", level = logger.LogLevel.INFO):
  logger.log(*values,sep=sep,end=end,file=file,level=level)

class EpsGreedy (ExplorationController):
    def __init__(self):
        super().__init__()
        self.base_config, _ = super().parse_args()
        self.config, _ = self.parse_args()
        self.epsilon = self.epsilon_init = self.config.initial_epsilon
        self.epsilon_delta = self.config.epsilon_delta
        self.eps_replay_factor = self.config.eps_replay_factor
        self.eps_min = self.config.final_epsilon
        self.last_dice = -1

        self.start_task = self.base_config.exploration_start_task
    
    def reset(self,factor=1.0):
        self.epsilon = self.epsilon_init * factor
        self.epsilon_delta = self.config.epsilon_delta * factor
        self.eps_min = self.config.final_epsilon ;

    def configure(self,curr_task,task):
        if task == self.start_task:
            self.reset() # reset epsilon to init
        elif curr_task > self.start_task: 
            self.reset(self.eps_replay_factor)  # NOTE: scale initial_eps & eps_delta for replay training.
        elif task < self.start_task: 
              log("----------------------BABBLING_, pure exploration!!") ;
              self.epsilon = 1.0 ;
              self.epsilon_delta = 0.0 ;
              self.eps_min = 1.0 ;
    
    def query(self):
        self.last_dice = np.random.random()
        return self.last_dice < self.epsilon and self.exploration

    def update(self):
        tmp = self.epsilon
        self.epsilon = self.epsilon - self.epsilon_delta if self.epsilon > self.eps_min else self.eps_min
        log("CHANGE EPSILON: ",tmp," - ",self.epsilon_delta," = ", self.epsilon)

    def update_epsilon(self): # THERE ARE 2 VERSIONS FOR SOME REASON... TODO: ONLY ALLOW ONE
        self.update()

    def get_current_status(self):
        return (round(self.epsilon,5),)
    
    def print_debug(self):
        log("---------EPSILON GREEDY---------")
        log("CURRENT: ",self.epsilon)
        log("DELTA  : ",self.epsilon_delta)
        log("MAX    : ",self.epsilon_init)
        log("MIN    : ",self.eps_min)
        log("--------------------------------")

    def define_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--initial_epsilon',          type=float, default=1.0,    help='The initial probability of choosing a random action.')
        parser.add_argument('--final_epsilon',            type=float, default=0.01,   help='The lowest probability of choosing a random action.')
        parser.add_argument('--epsilon_delta',            type=float, default=0.001,  help='Epsilon reduction factor (stepwise).')
        parser.add_argument('--eps_replay_factor',        type=float, default=0.5,    help='eps start for tasks > 0.')

    def parse_args(self):
        parser = argparse.ArgumentParser('EpsGreedy', 'argparser for EpsGreedy.', exit_on_error=False)
        self.define_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;
