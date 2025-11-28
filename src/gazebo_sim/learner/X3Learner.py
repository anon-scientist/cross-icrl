"""
X3Learner --- uses only two fixed actions for DCGMM. 0=go, 1=stop. RLAgent still assumes 4 actions, but these are internally mapped to 2 action for the DCGMM.
These are learned from vision. The direction of motion, however, is computed from laser data and thus not subject to learning.
HOW can we prevent that small cube views are learned by the DCGMM?
--> just sleep-learn the e.g. 5 or 10% biggest TD errors, then small cubes should be underrepresented
--> have lower bound on TD errors, below is not stored in buffer
"""

import math
import sys, os
import numpy as np
import tensorflow as tf ;

import time, random ;

import matplotlib
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

import argparse ;


from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, ReplayBufferTD, PrioritizedReplayBuffer ;
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models

from gazebo_sim.learner import ARLearner, BaseLearner ;
from gazebo_sim.model.GMM import build_model ;

from dcgmm.model import DCGMM
from dcgmm.callback import Set_Model_Params ;

from . import XLearner


class X3Learner(XLearner):

    """ here, we ignore n_actions and set it to 2!!"""
    def __init__(self, n_actions, obs_space, config):
      print("x3 init!!") ;
      XLearner.__init__(self, n_actions, obs_space, config) ;
      self.mode = "multimodal" ;
      self.n_actions = 2 ;

    ## same as X2Learner models but just with two outputs (stop, use laser)
    def build_models(self):
        self.n_actions = 2 ;
        # vanilla, model and target_model are the same. DQN and DDQN are not supported for da moment
        self.model = build_model("AR", self.input_dims, 2, self.config, ro_loss="q_learning") ;
        self.target_model = self.model ;
        self.gmm_opt = self.model.opt_and_layers[-2][0] ;
        self.ro_opt = self.model.opt_and_layers[-1][0] ;
        self.gmm_layer = self.model.layers[-2] ;
        self.ro_layer = self.model.layers[-1] ;
        self.frozen_model = build_model("AR-frozen", self.input_dims, 2, self.config, ro_loss="q_learning") ;
        self.frozen_gmm_layer = self.frozen_model.layers[-2] ;

        self.callback = Set_Model_Params(**vars(self.config)) ;



    # -------------------------------> action selection

    def merge_actions(self, laser_action, vis_action):
      if vis_action == 1: return 0
      else: return laser_action ;

    # DCGMM knows only 2 actions (0=go, 1=stop) but we translate this to four!
    # actions: 1 (left), 2(right), 3(straight), 0(stop)
    def choose_action(self, obs):
    
      if self.mode != "multimodal":
        return XLearner.choose_action(self, obs) ;

      # assume obs is a dictionary
      try:
        laser = obs["laser"] ;
        vis = np.array(obs["vis"]) ;
      except Exception:
        print("observations are not a dict") ;
        import sys;
        sys.exit(0) ;

      self.q_pred_all = self.invoke_model(vis[np.newaxis,:]) ;
      qs = self.q_pred_all ;

      nr_beams = laser.shape[0] ;
      laser_mask = np.logical_and(laser < 1.0, laser != np.inf).astype(np.float32) ;
      laser_cog = ( (np.arange(0,nr_beams,1)-nr_beams//2) * laser_mask ).sum() / laser_mask.sum() ;
      print("laser-cog: ", laser_cog) ;

      laser_action = 3 ;
      if laser_cog < -5: laser_action = 1 ;
      elif laser_cog > 5: laser_action = 2 ;

      # -------

      perform_babbling  = (self.task < self.config.exploration_start_task) ;
      perform_babbling = False ; # TMP!!

      if self.config.exploration == "eps-greedy":

        self.epsilon -= self.config.epsilon_delta ;
        # bound epsilon from below
        self.epsilon  = (self.config.final_epsilon if self.epsilon < self.config.final_epsilon else self.epsilon) ;

        rand = random.random() ;  
        if (rand < self.epsilon and self.exploitation_only == False) or perform_babbling == True :
          return self.merge_actions(laser_action, int(random.random()* 2)), True ;
        else:
          self.resp = tf.reduce_max(self.gmm_layer.get_output_result()) ;
          if self.resp < 0.6: # outlier
            return self.merge_actions(laser_action, 0), False; # outlier means go!!
          else:
            return self.merge_actions(laser_action, int(tf.argmax(tf.reshape(qs,-1)))), False ;

    def store_transition(self, state, action, reward, new_state, done):
        """ stores a transition in replay buffer, along with td error that is computed on the fly """

        ## remap action from 0,1,2,3 --> 0,1
        action = (0 if action in [0,1,2] else 1) ;
        XLearner.store_transition(self, state["vis"], action, reward, new_state["vis"], done) ;


      



