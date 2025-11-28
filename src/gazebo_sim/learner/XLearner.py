"""
XLearner: learning in cycles and selecting learned samples according to TD error.
TODO: Decoupled learning of representations and policies...
"""

import math
import sys, os
import numpy as np
import tensorflow as tf ;

import time ;

import matplotlib
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

import argparse ;


from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, ReplayBufferTD, PrioritizedReplayBuffer ;
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models

from gazebo_sim.learner import ARLearner ;
from gazebo_sim.model.GMM import build_model ;

from dcgmm.model import DCGMM
import gazebo_sim.utils.logger as logger

def log(*values:object, sep=' ', end='\n', file = "XLearner.txt", level = logger.LogLevel.INFO):
  logger.log(*values,sep=sep,end=end,file=file,level=level)

class XLearner(ARLearner):
    def __init__(self, n_actions, obs_space, config):
      ARLearner.__init__(self, n_actions, obs_space, config) ;

      if self.config.replay_buffer == "with_td":
        self.replay_buffer = ReplayBufferTD(self.config.capacity, self.input_dims)
      else:
        log("Obnly with_td buffer acceptable for X Learning!") ;
        sys.exit(-1) ;

      self.build_models() ;



# ---------------------------------------> BEFORE/AFTER
    def before_experiment(self):
      self.train_step = 0 ;

    def before_task(self, task):
        # reset exploration as a workaround. Should be replaced by TD-error scheme
        self.configure_exploration(task) ;
        self.train_step = -1 ;
        self.cycle_count = 0 ;

    def after_task(self, task):
        # do nothing because tasks are not importantr in this setting
        pass ;


    def store_transition(self, state, action, reward, new_state, done):
        """ stores a transition in replay buffer, along with td error that is computed on the fly """
        q_pred_all = self.invoke_model(state[np.newaxis,:]) ;
        resp = self.model.layers[-2].resp ;
        q_pred = q_pred_all[0,action] ;
        #log("Q", q_pred_all) ;
        q_next = self.invoke_model(new_state[np.newaxis,:])[0] ;
        q_obs = reward + self.config.gamma * tf.reduce_max(q_next) ;
        td_error = (q_pred - q_obs) ;
        abs_td_error = tf.math.abs(td_error) ;
        resp = self.model.layers[-2].resp ;
        log("TDERROR!!!", self.cycle_count, self.train_step, float(td_error), float(tf.reduce_max(resp))) ;
        self.replay_buffer.store_transition(state, action, reward, new_state, done, abs_td_error) ;


    def learn(self, task, curr_step):
        self.exploration.current_step_in_episode = curr_step
        self.train_step += 1 ;
        self.exploration.update_epsilon()

        # TODO create own counter, not the replay buffer one!!
        if self.replay_buffer.counter < self.config.train_batch_size: return ;

        ## --------- babbling --------------------
        if task < self.config.exploration_start_task:
            log("Babbling!", self.gmm_layer.tf_somSigma.numpy());
            states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.config.train_batch_size)
            td_error = self.learn_vanilla(task, states, actions, rewards, states_, terminal, learn_gmm = True, learn_ro = True)
            if self.train_step % 500 == 0:
              log("VIS/PROTOS") ;
              self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'babbling_{self.train_step}.png'), "current")
            return ;

        ## ---------wake phase ------------------------------------------------
        if self.train_step < self.config.cycle_length:
            log("wake!!", self.train_step, self.config.cycle_length) ;
            states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.config.train_batch_size)
            td_error = self.learn_vanilla(task, states, actions, rewards, states_, terminal, learn_gmm = False, learn_ro = True)
            return ;

        # -------- sleep phase ----------------------------------------------------

        # no sleep phase @ the end of a task
        if self.cycle_count == self.config.cycles_per_task-1: return ;

        # copy 2 frozen model & set train_step to 0 
        log("sleep") ;
        self.update_frozen_model(task) ; 
        self.replay_buffer.sort_td_wise() ;
        max_mem = self.replay_buffer.counter if self.replay_buffer.counter < self.replay_buffer.state_memory.shape[0] else self.replay_buffer.state_memory.shape[0] ;
        thr = (1-self.config.draw_percentage) * self.replay_buffer.td_memory[0] ; # determine td threshold from draw_percentage
        nr_samples_2_draw = (self.replay_buffer.td_memory > thr).astype("int32").sum() ; # how many sampels are above trheshold?
        draw_percentage = int(nr_samples_2_draw / self.replay_buffer.td_memory.shape[0]) # what fraction is that?
        log("Learning from", nr_samples_2_draw, thr, self.replay_buffer.td_memory[0]) ;
        np.savez(self.config.root_dir+"/results/"+self.config.exp_id+f"/buffer{task}_{self.cycle_count}.npz", self.replay_buffer.state_memory[0:max_mem]) ;


        for step in range(0,self.config.sleep_phase_train_steps):
          self.train_step += 1 ;
          states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.config.train_batch_size, self.config.draw_percentage)
          #states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.config.train_batch_size, draw_percentage)

          states = tf.convert_to_tensor(states, dtype=tf.float32)
          states_ = tf.convert_to_tensor(states_, dtype=tf.float32)

          # FIXME is this ok like that? AR should start in 2nd cycle ever, but not during a babbling task
          td_error = self.learn_vanilla(task, states, actions, rewards, states_, terminal, learn_gmm=True, learn_ro = True)

          if self.train_step % 500 == 0:
            log("VIS/PROTOS") ;
            self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'gmm_T{task}_{self.cycle_count}_{step}.png'), "current")


        # clear buffer
        self.replay_buffer.reset_buffer() ;
        self.cycle_count += 1 ;
        self.train_step = -1 ;





    # another way to have an extensible parser that re-uses the superclass parse_args()
    def parse_args(self):
        config,unparsed = ARLearner.parse_args(self) ;
        parser = argparse.ArgumentParser() ;
        parser.add_argument("--cycle_length", type=int, default=500, help="length of a learning cycle. d'uh!") ;
        parser.add_argument("--cycles_per_task", type=int, default=2, help="d'uh!") ;
        parser.add_argument("--sleep_phase_train_steps", type=int, default=1000, help="how many train steps in a cycle? d'uh!") ;
        parser.add_argument("--draw_percentage", type=float, default=0.5, help="what percentage of TD-sorted transitions to draw for training?") ;
        parser.add_argument('--replay_buffer',                 nargs='?', type=str,   default='default', choices=['with_td', 'default', 'prioritized'],  help='Replay buffer type to store experiences.')

        cfg2, unparsed = parser.parse_known_args() ;
        for (k,v) in cfg2.__dict__.items():
          setattr(config, k, v) ;

        return config, unparsed ;


