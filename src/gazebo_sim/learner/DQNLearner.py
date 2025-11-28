import math
import sys, os
import numpy as np
import tensorflow as tf ;

import matplotlib
import argparse ;
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

from gazebo_sim.exploration import EpsGreedy
from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models

from .BaseLearner import BaseLearner

class DQNLearner (BaseLearner):
    def __init__(self, n_actions, obs_space, config,**kwargs):
        BaseLearner.__init__(self, n_actions, obs_space, config,**kwargs) ;
        
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = self.config.train_batch_size

        self.gamma = self.config.gamma ;
        self.input_dims = obs_space

        self.exploration_controller = self.make_exploration_controller() ;
        #self.target_network_update_freq = self.config.target_network_upfate_freq ;

        self.babbling = True ;
        self.exploration_flag = True ;
        


    # -----------------------------------> PRE/POST ROUTINES

    def make_exploration_controller(self):
        return EpsGreedy() ;


    def build_models(self):
        self.n_actions = len(self.action_space) ;
        output_size = self.n_actions if self.config.output_size == -1 else self.config.output_size ;
        if self.config.dqn_dueling == "yes" and self.config.dqn_target_network == "yes": # dqn and dueling
            self.model, self.target_model = build_dueling_models(
                output_size,
                self.input_dims,
                self.config.dqn_fc1_dims,
                self.config.dqn_fc2_dims,
                float(self.config.dqn_adam_lr),
                self.config.train_batch_size
            ) ;
        elif self.config.dqn_dueling == "no" and self.config.dqn_target_network == "yes": # dqn but not dueling 
            print("!!!DDQN") ;
            self.model, self.target_model = build_dqn_models(
                output_size,
                self.input_dims,
                self.config.dqn_fc1_dims,
                self.config.dqn_fc2_dims,
                float(self.config.dqn_adam_lr),
                self.config.dqn_target_network) ;
            print(self.target_model) ;
        elif self.config.dqn_dueling == "yes" and self.config.dqn_target_network == "no": # dueling without dqn is not possible
          print("Dueling without dqn impossible") ;
          sys.exit(0) ;
        else: # vanilla, model and target_model are the same
            self.model, self.target_model = build_dqn_models(
                output_size,
                self.input_dims,
                self.config.dqn_fc1_dims,
                self.config.dqn_fc2_dims,
                float(self.config.dqn_adam_lr),
                self.config.dqn_target_network,
                "") ;
            self.target_model = self.model ;
        



# ---------------------------------------> BEFORE/AFTER
    ## Base class method
    def before_experiment(self):
        if self.config.replay_buffer == 'prioritized':
            self.replay_buffer = PrioritizedReplayBuffer(int(self.config.capacity), self.input_dims, self.config.per_alpha, self.config.per_beta, self.config.per_eps, self.config.per_delta_beta)
        else:
            self.replay_buffer = ReplayBuffer(int(self.config.capacity), self.input_dims)

        self.build_models() ;

    # To baseclass BaseLearner
    def episode_completed(self, incomplete=False): pass ;

    ## Base class method
    def before_task(self, task):
        self.train_step = 0 ;

        ## double dqn updating
        if self.config.dqn_target_network or self.config.dqn_dueling:
            self.update_model(force=True)  # force update when training is done!

        ## reset beta to its unannelaed value before each task
        if self.config.replay_buffer == "prioritized":
          self.replay_buffer.reset_beta() ;

        ##eps-greedy 
        self.get_exploration_controller().configure(self.task,task)

    ## Base class method
    def after_task(self, task):
        if task +1 == self.get_exploration_controller().start_task:
            print("-------------------------STOP-BABBLING,Resetting buffer!!")
            self.replay_buffer.reset()
        self.get_exploration_controller().print_debug()
        pass ;

    # -----------------------------------> BUFFER HANDLING

    ## Base class method
    def store_transition(self, state, action, reward, new_state, done):
        """ stores a transition in replay buffer and optionally saves buffer content to file for offline pre-training. """
        self.replay_buffer.store_transition(state, action, reward, new_state, done)

    ## Base class method
    def invoke_model(self, data): # always batches never individual samples
      return self.model(data) ;

    def invoke_target_model(self, data): # always batches never individual samples
      return self.target_model(data) ;

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


    def post_process_buffer(self, states, actions, rewards, states_, terminal):
      pass ;
        

    ## Base class method
    def learn(self, task, curr_step):
        self.get_exploration_controller().current_step_in_episode = curr_step
        self.train_step += 1 ;
        if self.train_step < self.batch_size:
            return  # if buffer is not full yet -> pass
        
        # FIXME: ugh, but2lazy
        #if self.config.algorithm == 'QGMM':
        #    if self.algorithm.gmm_training_active and self.epsilon == self.eps_min:
        #        self.algorithm.gmm_training_active = False  # NOTE: disable GMM layer training when eps_min is reached

        weights = 1.0 ;
        if self.config.replay_buffer == 'prioritized':
          # extract priorities
          states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.batch_size)
          weights = self.replay_buffer.get_weights_current_batch() ;
        else:
          states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.batch_size)

        self.post_process_buffer(states, actions, rewards, states_, terminal) ;


        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)

        if self.config.dqn_target_network == "yes" and self.config.dqn_dueling == "yes": ## 3dqn
            td_error = self.learn_doubleq(states, actions, rewards, states_, terminal, weights=weights)
        else: # DDQN, DQN
            td_error = self.learn_vanilla(states, actions, rewards, states_, terminal, weights=weights)

        if self.config.replay_buffer == 'prioritized':
            self.replay_buffer.update_priorities(batch_indices, td_error)
            
        self.get_exploration_controller().update_epsilon()
        #OLDself.exploration.update()

    # dqn and ddqn, weights should be 1d-ndarray 
    def learn_vanilla(self, states, actions, rewards, states_, dones, weights=1.0):
        dqn_variables = self.model.trainable_variables
        with tf.GradientTape() as g:
            # Get Q values for current obs s_{t} using online model: Q(s, a, theta_i)
            online_q_cur = self.model(states)
            pred_q_values = tf.gather(online_q_cur, actions, batch_dims=1)
            # ------------------------------------------------------------------------>
            target_q_next_qs = tf.stop_gradient(self.target_model(states_))
            # bellman equation
            target_q_values = self.bellman(target_q_next_qs, rewards, dones)
             # ------------------------------------------------------------------------>
            td_error = tf.square(pred_q_values - target_q_values)
            loss = tf.reduce_mean(td_error * weights)
            gradients = g.gradient(loss, dqn_variables)
            self.model.optimizer.apply_gradients(zip(gradients, dqn_variables))
        del g
        self.update_model()
        return td_error ;

    # dueling
    def learn_doubleq(self, states, actions, rewards, states_, dones, weights = None):
        # Q(s,a;θ) = r + γQ(s', argmax_{a'}Q(s',a';θ);θ')
        # θ: online net; θ': frozen (target) network
        # θ decides best next action a'; θ' evaluates action (Q-value estimation)
        dqn_variables = self.model.trainable_variables
        td_error = None ;
        with tf.GradientTape() as g:
            # Get Q values for current obs s_{t} using online model: Q(s, a, theta_i)
            online_q_cur = self.model(states)
            pred_q_values = tf.gather(online_q_cur, actions, batch_dims=1)
            # ------------------------------------------------------------------------>
            # Get Q values for best actions in next state s_{t+1} using online model: max(Q(s', a', theta_i)) w.r.t a'
            online_q_next = tf.stop_gradient(self.model(states_))
            online_q_next_max = tf.argmax(online_q_next, axis=-1)
            online_q_next_action_mask = tf.one_hot(online_q_next_max, self.n_actions) # one hot mask for actions
            # ------------------------------------------------------------------------>
            target_q_next = tf.stop_gradient(self.target_model(states_))            
            # Get Q values from target network for next state s_{t+1} and chosen action
            self.target_q_next_qs = tf.reduce_sum(online_q_next_action_mask * target_q_next, axis=-1)
            # bellman equation
            self.target_q_values = rewards + self.gamma * self.target_q_next_qs * dones
             # ------------------------------------------------------------------------>
            td_error = tf.square(pred_q_values - self.target_q_values)
            loss = tf.reduce_mean(td_error)
            gradients = g.gradient(loss, dqn_variables)
            self.model.optimizer.apply_gradients(zip(gradients, dqn_variables))   
        del g

        self.update_model()
        return td_error ;

    def bellman(self, q_next, rewards, dones):
        """ bellman equation """
        max_next_q_values = tf.reduce_max(q_next, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values * dones
        return target_q_values

    def update_model(self, force=False):
        if self.config.dqn_target_network != "yes": return ;
        if self.train_step == 0 and not force: return
        if (self.train_step % self.config.dqn_target_network_update_freq == 0) or force:
            self.copy_model_weights(self.model, self.target_model)

        


    def copy_model_weights(self, source, target):
        ''' in-memory copy of model weights '''
        
        for source_layer, target_layer in zip(source.layers, target.layers):
            source_weights = source_layer.get_weights()
            #print("SOURCE", source_weights)
            target_layer.set_weights(source_weights)
            target_weights = target_layer.get_weights()

            #if source_weights and all(tf.nest.map_structure(np.array_equal, source_weights, target_weights)):
            #    print(f'\033[93m[INFO]\033[0m [QGMM]: WEIGHT TRANSFER: {source.name}-{source_layer.name} -> {target.name}-{target_layer.name}')
        print(f'\033[93m[INFO]\033[0m : WEIGHT TRANSFER: {source.name}-->{target.name}')

    def define_base_args(self, parser):
        """ parser is a KwargParser instance with kwargs alreay stored. Created by constructor of BaseLearner """
        # ------------------------------------ LEARNER
        parser.add_argument('--start_task',                  type=int, default=0        ,          help='port for TCP debug connections')
        parser.add_argument('--seed',     type=int, default=42,                          help='The random seed for the experiment run.')
        parser.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated results.')
        parser.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')

        parser.add_argument('--debug',                   type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')

        parser.add_argument('--train_batch_size',       type=int,   default=32,      help='Defines the mini-batch size that is used for training.')
        parser.add_argument('--train_batch_iteration',  type=int,   default=1,       help='Defines how often the mini-batch is used for training.')
        parser.add_argument('--gamma',                  type=float, default=0.95,    help='The discount factor of the bellman equation.')
        parser.add_argument('--exploration',            type=str, default="eps-greedy", choices=['eps-greedy','eps-list','eps-protos'], help='The exploration strategy the agent should use.')

        parser.add_argument('--replay_buffer',                 nargs='?', type=str,   default='default', choices=['default', 'prioritized','with_td'],  help='Replay buffer type to store experiences.')
        parser.add_argument('--capacity',                                 type=int,   default=1000,                                           help='Buffer storage capacity.')
        parser.add_argument('--output_size',                              type=int,   default=-1,                                           help='Size of mode loutput layer, use n_actions for a value of -1')
        parser.add_argument('--per_alpha',                                type=float, default=0.6,                                            help='Sets the degree of prioritization used by the buffer [0, 1].')
        parser.add_argument('--per_beta',                                 type=float, default=0.4,                                            help='Sets the degree of importance sampling to suppress the influence of gradient updates [0, 1].')
        parser.add_argument('--per_eps',                                  type=float, default=1e-06,                                          help='Epsilon to add to the TD errors when updating priorities.')
        parser.add_argument('--per_delta_beta',                           type=float, default=1e-04,                                          help='Controls linear annealing of beta') ;        

        parser.add_argument('--model_type', type=str, default='DNN', choices=['DNN', 'CNN'],   help='Sets the model architecture for the learner backend.')

        parser.add_argument('--dqn_fc1_dims',                    type=int,   default=128,                                    help='Size of FC layer 1.')
        parser.add_argument('--dqn_fc2_dims',                    type=int,   default=64,                                     help='Size of FC layer 2.')
        parser.add_argument('--dqn_adam_lr',                     type=float, default=1e-3,                                   help='Learning rate for ADAM opt.')
        parser.add_argument('--dqn_dueling',                     type=str,   default='no',        choices=['yes', 'no'],     help='Use dueling DQNs?')
        parser.add_argument('--dqn_target_network',              type=str,   default='no',        choices=['yes', 'no'],     help='Whether to use double DQN (target network).')
        parser.add_argument('--dqn_target_network_update_freq',  type=int,   default=1000,                                   help='Sets the number of steps after which the target model is updated.')

