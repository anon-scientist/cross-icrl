"""
TODO:
- all interface methods should get 'task' param?
- a self.exploration attribute is not a given!
"""

import math
import sys, os
import numpy as np
import tensorflow as tf ;

import matplotlib
import argparse ;
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

from gazebo_sim.exploration import ExplorationController ;
from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models
import gazebo_sim.utils.logger as logger

from cl_experiment.parsing import Kwarg_Parser ;

def log(*values:object, sep=' ', end='\n', file = "BaseLearner.txt", level = logger.LogLevel.INFO):
  logger.log(*values,sep=sep,end=end,file=file,level=level)

class BaseLearner:
    def __init__(self, n_actions, obs_space, config, **kwargs):
        self.config, _ = self.parse_args(**kwargs) ;
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = self.config.train_batch_size

        self.gamma = self.config.gamma ;
        self.input_dims = obs_space ;

        self.exploration_controller = self.make_exploration_controller() ;

        self.babbling = True ;
        self.exploration_flag = True ;


    # -----------------------------------> PRE/POST ROUTINES

    def build_models(self):
      pass ;


# ---------------------------------------> BEFORE/AFTER task or experiment
    def before_experiment(self):
      pass ;

    def before_task(self, task):
      pass ;

    def after_task(self, task):
      pass ;

    def episode_completed(self, incomplete=False): 
      pass ;

    # -----------------------------------> BUFFER HANDLING

    def store_transition(self, state, action, reward, new_state, done):
      """ stores a transition in replay buffer and optionally saves buffer content to file for offline pre-training. """
      pass ;

    # ----------------------------------> ACTION SELECTION

    # completely model agnostic as long as model is keras model
    def choose_action(self, observation):
        randomly_chosen = self.get_exploration_controller().query() ;
        if randomly_chosen:
            #print("actions pacve", len(self.action_space)) ;
            action = np.random.randint(0, len(self.action_space))
        else:
            #print("Q-Values", actions) ;
            state = observation
            actions = self.invoke_model(state[np.newaxis,:]) ;
            #print(observation[0], observation[1], observation[2])
            #print("QQQQQ", actions[0]) ;
            action = int(np.argmax(actions, axis=1))
        return action, randomly_chosen

    # --------------------------------> LEARNING

    ## TODO: curr_step?
    def learn(self, task, curr_step):
      pass ;

    def bellman(self, q_next, rewards, dones):
        """ bellman equation """
        max_next_q_values = tf.reduce_max(q_next, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values * dones
        return target_q_values

    # -------------------------------> EXPLORATION

    # overwrite
    def make_exploration_controller(self):
      return ExplorationController() ;

    def get_exploration_controller(self):
      return self.exploration_controller ;

    def enable_babbling(self): # pure exploration
      self.exploration_controller.enable_babbling() ;
      print("+++++++++++++++++++++Babbling wenabled!") ;

    def disable_babbling(self): # pure exploration
      self.exploration_controller.disable_babbling() ;
      print("+++++++++++++++++++++Babbling disabled!") ;

    def disable_exploration(self):
      self.exploration_controller.disable_exploration();

    def enable_exploration(self):
      self.exploration_controller.enable_exploration();


    # -------------------------------> COMMUNICATION WITH AGENT

    def get_current_status(self):
      return self.exploration_controller.get_current_status() ;


    def set_task(self, task):
      self.task = task ;

    # -------------------------------> HELPERS

    def load(self, ckpt): 
      log("Loading", ckpt) ;
      self.model.load_weights(ckpt) ;
  
    def save(self, ckpt): 
      self.model.save_weights(ckpt) ;

    def invoke_model(self, data): # always batches never individual samples
      return self.model(data) ; # return model result as naked TF2 tensor

    def invoke_target_model(self, data): # always batches never individual samples
      return self.model(data) ;

    """
    def copy_model_weights(self, source, target):
        ''' in-memory copy of model weights '''
        
        for source_layer, target_layer in zip(source.layers, target.layers):
            source_weights = source_layer.get_weights()
            #log("SOURCE", source_weights)
            target_layer.set_weights(source_weights)
            target_weights = target_layer.get_weights()

            #if source_weights and all(tf.nest.map_structure(np.array_equal, source_weights, target_weights)):
            #    log(f'\033[93m[INFO]\033[0m [QGMM]: WEIGHT TRANSFER: {source.name}-{source_layer.name} -> {target.name}-{target_layer.name}')
        log(f'\033[93m[INFO]\033[0m [QGMM]: WEIGHT TRANSFER: {source.name}-->{target.name}')
    """
    # -----------------------------------------> ARGUMENT PARSING

    def define_base_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--start_task',                  type=int, default=0        ,          help='port for TCP debug connections')
        parser.add_argument('--seed',     type=int, default=42,                          help='The random seed for the experiment run.')
        parser.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated results.')
        parser.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')

        parser.add_argument('--debug',                   type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')

        parser.add_argument('--train_batch_size',       type=int,   default=32,      help='Defines the mini-batch size that is used for training.')
        parser.add_argument('--train_batch_iteration',  type=int,   default=1,       help='Defines how often the mini-batch is used for training.')
        parser.add_argument('--exploration',     type=str, default="eps-greedy", help='The exploration strategy the agent should use.')

        parser.add_argument('--model_type', type=str, default='DNN', choices=['DNN', 'CNN'],   help='Sets the model architecture for the learner backend.')


    def parse_args(self, **kwargs):
        #parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.', exit_on_error=False)
        parser = Kwarg_Parser(**kwargs) ;
        self.define_base_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;

