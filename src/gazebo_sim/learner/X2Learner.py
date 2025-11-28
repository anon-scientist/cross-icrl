"""
XLearner: learning in cycles and selecting learned samples according to TD error.
TODO: Decoupled learning of representations and policies...
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

from gazebo_sim.exploration.EpsilonGreedy import EpsGreedy
from gazebo_sim.exploration.EpsilonProtos import EpsProtosGreedy as EpsProtos

from gazebo_sim.learner import ARLearner, BaseLearner ;
from gazebo_sim.model.GMM import build_model ;

from dcgmm.model import DCGMM
from dcgmm.callback import Set_Model_Params ;

from .BaseLearner import BaseLearner
import gazebo_sim.utils.logger as logger

def log(*values:object, sep=' ', end='\n', file = "X2Learner.txt", level = logger.LogLevel.INFO):
  logger.log(*values,sep=sep,end=end,file=file,level=level)

class XLearner(BaseLearner):
    def __init__(self, n_actions, obs_space, config):
      BaseLearner.__init__(self, n_actions, obs_space, config) ;
      self.n_actions = n_actions ;
      self.gamma = self.config.gamma ;

      if self.config.exploration == "eps-greedy":
        self.exploration = EpsGreedy() ;
      elif self.config.exploration == "eps-protos":
        self.exploration = EpsProtos(self.config.qgmm_K) ;
      else:
        log("No Valid Exploration Strategy Chosen!")
        sys.exit(-1) ;

      if self.config.replay_buffer == "with_td":
        self.replay_buffer = ReplayBufferTD(self.config.capacity, self.input_dims)
      else:
        log("Obnly with_td buffer acceptable for X Learning!") ;
        sys.exit(-1) ;

      self.build_models() ;

    def build_models(self):
        self.n_actions = len(self.action_space) ;
        # vanilla, model and target_model are the same. DQN and DDQN are not supported for da moment
        self.model = build_model("AR", self.input_dims, len(self.action_space), self.config, ro_loss="q_learning") ;
        self.target_model = self.model ;
        self.gmm_opt = self.model.opt_and_layers[-2][0] ;
        self.ro_opt = self.model.opt_and_layers[-1][0] ;
        self.gmm_layer = self.model.layers[-2] ;
        self.ro_layer = self.model.layers[-1] ;
        self.frozen_model = build_model("AR-frozen", self.input_dims, len(self.action_space), self.config, ro_loss="q_learning") ;
        self.frozen_gmm_layer = self.frozen_model.layers[-2] ;

        self.callback = Set_Model_Params(**vars(self.config)) ;




# ---------------------------------------> BEFORE/AFTER
    def before_experiment(self):
      self.train_step = 0 ;

    def before_task(self, task):
        # reset exploration as a workaround. Should be replaced by TD-error scheme
        #self.configure_exploration(task) ;
        self.train_step = -1 ;
        self.cycle_count = 0 ;
        self.exploration.configure(self.current_task,task)
        self.current_task = task ;
        self.sleep_phases = 0 ;

    def after_task(self, task):
        # do nothing because tasks are not importantr in this setting
        pass ;

    # -------------------------------> exploration
    def set_babbling(self, flag):# THIS FLAG IS NOT NEEDED -> TAKEN CARE OF IN EXPLORATION CLASS
      self.babbling = flag ;

    def enable_exploration(self):
      self.exploration.enable_exploration()

    def disable_exploration(self):
      self.exploration.disable_exploration()

    def get_exploration_status(self):
      return (not self.exploration.exploration) ;

    def get_current_status(self):
      return self.exploration.get_current_status() ;


    # -------------------------------> action selection

    def dist_to_stm(self, obs):
      return 0.0 ;

    def choose_action(self, obs):
      self.q_pred_all = self.invoke_model(obs[np.newaxis,:]) ;
      self.resp = self.gmm_layer.get_output_result() ;

      if self.exploration.name == "eps-proto": # or self.config.exploration == "eps-proto"
        flat_resp = self.resp.numpy().reshape(-1)
        self.exploration.current_proto = np.argmax(flat_resp)
        self.exploration.proto_certainty = flat_resp[np.argmax(flat_resp)]

      #if self.config.exploration == "curiosity":
#
      #  if ((tf.reduce_max(self.resp) < self.dist_to_stm(obs)) and self.exploitation_only == False) or perform_babbling == True:
      #    action = random.randint(0,self.n_actions) ;
      #    self.performed_exploration = True ;
      #    return action, True ;
      #  else:
      #    self.performed_exploration = False ;
      #    action = tf.argmax(self.model(tf.convert_to_tensor([obs]))[0]) ;
      #    return action, False ;

      self.exploration.update() # Updates the exploration policy (e.g. by decreasing the epsilon)
      if self.exploration.query():
        return int(random.random()* self.n_actions), True ;
      else:
        return int(tf.argmax(self.q_pred_all[0])), False ;


    def store_transition(self, state, action, reward, new_state, done):
        """ stores a transition in replay buffer, along with td error that is computed on the fly """
        # compute q_pred from model output from call in choose_action
        q_pred = self.q_pred_all[0,action] ;
        #log("Q", q_pred_all) ;
        q_next = self.invoke_model(new_state[np.newaxis,:])[0] ;
        q_obs = reward + self.config.gamma * tf.reduce_max(q_next) ;
        td_error = (q_pred - q_obs) ;
        abs_td_error = tf.math.abs(td_error) ;
        resp = self.gmm_layer.get_output_result() ;
        log("TDERROR!!!", self.cycle_count, self.train_step, float(td_error), float(tf.reduce_max(resp))) ;

        if self.config.exploration == "curiosity":
          # fill buffer if TD error is problematic for exploitation
          if self.performed_exploration == False:
            stm_threshold = 0.8 ;
            if abs_td_error > stm_threshold: self.add_to_stm(state) ;
          else:
            self.replay_buffer.store_transition(state, action, reward, new_state, done, abs_td_error) ;

        elif self.exploration.name == "eps-greedy" or self.exploration.name == "eps-protos":
          self.replay_buffer.store_transition(state, action, reward, new_state, done, abs_td_error) ;


    """
    AR core functionality: given a set of observations ('samples'), we generate 'ratio'  samples
    using the frozen model. These, we filter for outliers (RIL mechanism).
    For each generated sample, we create pseudo-labels using the frozen model
    If wake phase: real model ok for RIL, frozen model for pseudo-labels. Because: was copied @ start of wake phase, is up to date
    if sleep phase: real model for RIL, real model nedded for pseudo-labels. Because: real
    """
    def generate_samples(self, states):
        # check which samples in 'states' are outliers
        self.frozen_model(states) ; # wake phase: frozen model, clear, gmm part is same as model. For sleep phase: frozen model, too since it has just been copied. So, always FM!
        scores = tf.reduce_max(self.frozen_gmm_layer.get_output_result(), axis=(1,2,3)) ; # N,
        # mask has 'True' for inlier samples
        mask = tf.greater(scores, self.config.ril_inlier_threshold) ;
        log("mask shape ", mask.shape) ;

        ratio = self.config.ar_replay_ratio ;

        # generate inlier-filtered batches
        gen_batches = [tf.boolean_mask(self.frozen_model.do_variant_generation(
            states, selection_layer_index=1, activity_layer_index=2), mask) for i in range(0,ratio) ] ;
        gen_obs = tf.concat(gen_batches, axis=0) ;

        # generate pseudo-labels, format N,C, where only one entry is != 0, depe3nding on a random action
        outputs_gen = self.frozen_model(gen_obs) ; # N,C (# classes)

        # convert pseudo-labels to "one-hot" needed for the Readout_Layer of DCGMM
        # TODO: use tf random numbers!
        C = self.n_actions ;
        random_actions = tf.convert_to_tensor(np.random.randint(0,C,size=(outputs_gen.shape[0],))) ; # N,
        output_mask = tf.gather(tf.eye(C),random_actions) ;
        gen_targets = outputs_gen * output_mask ;

        return gen_obs, gen_targets ;



    # sleep pphase learning with graph mode DCGMM using fit()
    # either to be used for  babbling or for sleep phase. Governed by parameter draw_percentage
    # For sleep phases, do NOT create samples on the fly, but for whole dataset from buffer!
    # for babbling, to no generate sampels at all but just use buffer
    def sleep_phase(self, task, draw_percentage):
        self.model.reset() ;
        self.update_frozen_model(task) ;
        log("Sleep with ", self.gmm_layer.tf_somSigma) ;

        # extract x % of highest-td samples from buffer
        self.replay_buffer.sort_td_wise() ;
        max_mem = self.replay_buffer.getNrEntries() ;
        max_idx = int(max_mem * draw_percentage) ;
        #real_obs = self.replay_buffer.state_memory[0:max_idx] ;
        #real_next_obs = self.replay_buffer.new_state_memory[0:max_idx] ;
        #real_actions = self.replay_buffer.action_memory[0:max_idx] ;
        #real_dones = self.replay_buffer.terminal_memory[0:max_idx] ;
        #real_rewards = self.replay_buffer.reward_memory[0:max_idx] ;
        real_obs, real_actions, real_rewards, real_next_obs, real_dones, _ = self.replay_buffer.sample_buffer(batch_size = int(self.replay_buffer.getNrEntries() * draw_percentage), td_percentage=draw_percentage,replace=False) ;
        C = self.n_actions ;
        log("Sleep: buffer has %d entries and we draw %d of them" % (max_mem, max_idx)) ;
        np.savez(self.config.root_dir+"/results/"+self.config.exp_id+f"/maxTD{task}_{self.train_step}.npz", real_obs) ;
        np.savez(self.config.root_dir+"/results/"+self.config.exp_id+f"/buffer{task}_{self.train_step}.npz", self.replay_buffer.state_memory[0:max_mem]) ;

        # TODO it is reasonable to use q's here? maybe only rewards instead of bellman derived from model that is static?
        real_targets = self.bellman(self.model(real_next_obs),real_rewards,real_dones) ; # N,1
        log(real_targets.shape);
        real_targets = tf.gather(tf.eye(C),real_actions) * tf.expand_dims(real_targets,1);
        log(real_targets.shape);
        

        # generate hallucinated samples and targets here!
        B = 32 ; # TODO: make equal to batch size
        ratio = 1 ;
        gen_obs = [] ;
        gen_targets = [] ;
        log("max_idx", max_idx) ;
        for i in range(0,max_idx // B):
            states = tf.convert_to_tensor(real_obs[i*B:(i+1)*B], dtype=tf.float32) ;
            actions = tf.convert_to_tensor(real_actions[i*B:(i+1)*B], dtype=tf.int32)  ;
            log("Generting", i, states.shape, actions.shape) ;
            _obs, _targets = self.generate_samples(states) ;
            gen_obs.append(_obs) ; gen_targets.append(_targets) ;

        log([g.shape for g in gen_obs], "GENOBS")
        log([t.shape for t in gen_targets], "targets")

        X_gen = tf.concat(gen_obs,axis=0) ;
        np.savez(self.config.root_dir+"/results/"+self.config.exp_id+f"/gen.npz", X_gen)
        T_gen = tf.concat(gen_targets,axis=0) ;
        X = tf.concat([X_gen, real_obs], axis=0) ;
        T = tf.concat([T_gen, real_targets], axis=0) ;

        self.gmm_layer.active = True ;
        self.ro_layer.active = True ;
        self.ro_layer.pre_train_step()

        self.model.fit(X,T,shuffle=True, epochs = self.config.sleep_phase_epochs, callbacks = [self.callback]) ;

        self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'sleepT{task}-{self.sleep_phases}.png'),"current") ;
        self.sleep_phases += 1 ;



    def learn(self, task, curr_step=None):
        """
        Governs babbling(leanr_gmm = learn_ro = True), wake cycle(learn_gmm=False, learn_ro = True) and sleep phases (learnin_gmm = True)
        Decision based on task (only for babbling phase) and train_step (reset @ start of each task). 
        A babbling task is created in RLAgent, outside the control of learner.
        This is why we use task and reset train_step @ start of task.  Normally train_step should not be reset, 
        and learn would not use 'task' at all.
        """
        self.train_step += 1 ;
        log(self.gmm_layer.tf_somSigma, "SIG") ;

        # TODO: babbling offline,too?
        ## --------- babbling/wake phase ------------------------------------------------
        if (self.train_step % self.config.wake_phase_length) < self.config.wake_phase_length-1:
            babblingPhase = (task < self.config.exploration_start_task ) ;
            #print("babbling" if babblingPhase == True else "wake_phase!", self.train_step, self.config.wake_phase_length) ;
            if self.train_step == 0 and task == 0: self.gmm_layer.tf_somSigma.assign(4.0) ; # SOM-sigma reset

            # reset buffer @ start of cycle
            if self.train_step % self.config.wake_phase_length == 0: 
              self.exploration.reset()
              log("clear buffer!") ; 
              self.replay_buffer.reset_buffer() ;
              # copy 2 frozen model
              self.update_frozen_model(task) ;

            # no training before replay buffer is not sufficiently filled!
            if self.replay_buffer.counter < self.config.train_batch_size: log("Buffer not filled yet!") ; return ;

            # train gmm + ro for babbling, or just ro for wake phase
            states, actions, rewards, states_, terminal, batch_indices = self.replay_buffer.sample_buffer(self.config.train_batch_size)
            td_error = self.wake_phase_learning(task, states, actions, rewards, states_, terminal, train_gmm = (task < self.config.exploration_start_task ))

            if self.train_step % 500 == 0 and babblingPhase:
                print("VIS/PROTOS") ;
                self.vis_protos(os.path.join(self.config.root_dir, 'results', self.config.exp_id, f'babbling_{task}_{self.train_step}.png'), "current")

            return ;

        # -------- sleep phase ----------------------------------------------------

        if (task > 0 ) and (self.train_step % self.config.wake_phase_length == self.config.wake_phase_length-1):
          log("Sleep phase") ;
          self.sleep_phase(task, self.config.draw_percentage) ;
          # clear buffer after sleep phase
          # TODO: clear STM as well!!
          self.cycle_count += 1 ;
          self.replay_buffer.reset_buffer() ;


    # simple Q LEARN, no ddqn for the moment!
    def wake_phase_learning(self, task, states, actions, rewards, nextStates, dones, weights=1.0, train_gmm = False):
        #log(states.min(), states.max()) ;
        #log("F", self.ro_layer.get_grad_factors())
        t_start = time.time_ns()
        #log("TASK!!!", task, self.train_step) ;
        log(task, "traingm,m=", train_gmm) ;
        ratio = self.config.ar_replay_ratio ;

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        nextStates = tf.convert_to_tensor(nextStates, dtype=tf.float32)
        C = self.n_actions ;

        # GENERATE real TARGETS use the Bellman eqn
        pred_real_next = self.invoke_target_model(nextStates) # 32,6
        q_real = self.bellman(pred_real_next, rewards, dones) # N,1
        targets = tf.expand_dims(q_real,1) * tf.gather(tf.eye(C), actions) ;

        # generate samples and targets from frozen model. Filter them fopr inliers and generate only from inlier samples!
        if task >= self.config.start_task_ar:
            states_gen,targets_gen = self.generate_samples(states) ;
            targets = tf.concat([targets, targets_gen], axis=0) ;
            states=tf.concat([states,states_gen], axis=0)

        self.ro_layer.set_active(True) ;
        self.gmm_layer.set_active(train_gmm) ;
        self.gmm_layer.pre_train_step() ;
        self.ro_layer.pre_train_step()
        loss = self.model.train_step((states, targets))
        self.gmm_layer.post_train_step() ;
        self.ro_layer.post_train_step() ;

        return loss ;

    def update_frozen_model(self, task):
          log("--------------------COPYING!!") ;
          #self.model.reset() ;
          self.copy_model_weights(self.model, self.frozen_model) ;
          self.frozen_model.layers[-2].tf_somSigma.assign (0.6) ; # TODO: param


    # another way to have an extensible parser that re-uses the superclass parse_args()
    def parse_args(self):
        config,unparsed = BaseLearner.parse_args(self) ;
        parser = argparse.ArgumentParser() ;
        parser.add_argument('--exploration',            type=str, default=None, choices=['eps-greedy','eps-list','eps-protos'], help='The exploration strategy the agent should use.')
        parser.add_argument('--exploration_start_task',   type=int, default=0   ,     help='all taskl before this one are babbling')
        parser.add_argument("--wake_phase_length", type=int, default=500, help="length of a learning cycle. duh!") ;
        parser.add_argument("--start_task_ar", type=int, default=1, help="start for AR. Duh!") ;
        parser.add_argument("--sleep_phase_epochs", type=int, default=1000, help="how many train steps in a cycle? duh!") ;
        parser.add_argument("--draw_percentage", type=float, default=0.3, help="what percentage of TD-sorted transitions to draw for training?") ;
        parser.add_argument("--gamma", type=float, default=0.9, help="discount factor") ;
        parser.add_argument('--replay_buffer',                 nargs='?', type=str,   default='default', choices=['with_td', 'default', 'prioritized'],  help='Replay buffer type to store experiences.')
        parser.add_argument("--capacity", type=int, default=1000, help="replay buffer size") ;
        parser.add_argument('--ar_replay_ratio',                type=int,   default=2,                                     help='Ratio between generated and real samples')
        parser.add_argument('--ril_inlier_threshold',           type=float, default=0.0,                                    help='Filter samples whose score smaller than this threshold ') ;
        # ----------------
        parser.add_argument('--qgmm_K',                         type=int,   default=100,                                     help='Number of K components to use for the GMM layer.')
        parser.add_argument('--qgmm_eps_0',                     type=float, default=0.011,                                  help='Start epsilon value (initial learning rate).')
        parser.add_argument('--qgmm_eps_inf',                   type=float, default=0.01,                                   help='Smallest epsilon value (learning rate) for regularizatiopm') ;
        parser.add_argument('--qgmm_lambda_sigma',              type=float, default=0.,                                     help='Sigma factor.')
        parser.add_argument('--qgmm_lambda_pi',                 type=float, default=0.,                                     help='Pis factor.')
        parser.add_argument('--qgmm_alpha',                     type=float, default=0.01,                                   help='Regularizer alpha.')
        parser.add_argument('--qgmm_gamma',                     type=float, default=0.9,                                    help='Regularizer gamma.')
        parser.add_argument('--qgmm_regEps',                    type=float, default=0.01,                                   help='Learning rate for the readout layer (SGD epsilon).')
        parser.add_argument('--qgmm_lambda_W',                  type=float, default=1.,                                     help='Weight factor.')
        parser.add_argument('--qgmm_lambda_b',                  type=float, default=0.,                                     help='Bias factor.')
        parser.add_argument('--qgmm_reset_somSigma',            type=float, default=2.5,                      help='Resetting annealing radius which is set with each new sub-task.')
        parser.add_argument('--qgmm_somSigma_sampling',         type=str,   default='yes',        choices=['yes', 'no'],    help='Activate to uniformly sample from a radius around the B') ;
        parser.add_argument('--qgmm_log_protos_each_n',         type=int,   default=0,                                      help='Saves protos (as an image) each N steps.')
        parser.add_argument('--qgmm_load_ckpt',                 type=str,   default="no",choices=["yes","no"],                                            help='Provide a path to a check') ;
        parser.add_argument('--qgmm_init_forward',              type=str,   default='yes',        choices=['yes', 'no'],    help='Init weights to favor driving forward.')
        # --------------------------
        parser.add_argument('--exploration',   type=str, default="eps-greedy"   ,     help='duh')
        parser.add_argument('--exploration_start_task',   type=int, default=0   ,     help='all taskl before this one are babbling')
        parser.add_argument('--initial_epsilon',          type=float, default=1.0,    help='The initial probability of choosing a random action.')
        parser.add_argument('--final_epsilon',            type=float, default=0.01,   help='The lowest probability of choosing a random action.')
        parser.add_argument('--epsilon_delta',            type=float, default=0.001,  help='Epsilon reduction factor (stepwise).')
        parser.add_argument('--eps_replay_factor',        type=float, default=0.5,    help='eps start for tasks > 0.')
        # ----------------


        cfg2, unparsed = parser.parse_known_args() ;
        for (k,v) in cfg2.__dict__.items():
          setattr(config, k, v) ;

        return config, unparsed ;

    def vis_protos(self, name, which_model):
        K = self.gmm_layer.K
        c = self.gmm_layer.c_in
        mus = self.gmm_layer.mus
        if which_model == "frozen":
          mus = self.frozen_model.layers[-2].mus ;
        mus = mus.numpy()
        mus = mus.reshape(K, c)
        f, axes = plt.subplots(int(math.sqrt(K)), int(math.sqrt(K)))
        f.set_figwidth(int(math.sqrt(K))) ;
        f.set_figheight(int(math.sqrt(K))) ;

        for i, ax in enumerate(axes.ravel()):
            ax.imshow(mus[i].reshape(self.input_dims))
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
        #plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.01, right=0.99,top=0.99,bottom=0.01)
        plt.savefig(name, bbox_inches='tight')
        plt.close('all')





