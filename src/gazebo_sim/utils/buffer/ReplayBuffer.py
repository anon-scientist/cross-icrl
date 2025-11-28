import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, input_dims):
        self.buffer_size = buffer_size
        self.input_dims = input_dims
        self.reset()

    def reset(self):
        self.counter = 0
        self.state_memory = np.zeros((self.buffer_size, *self.input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.buffer_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.buffer_size,dtype=np.int32)

    def store_transition(self,state, action, reward, state_, done):
        index = self.counter % self.buffer_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal, batch
    
    def update_priorities(self, indices, td_errors):
        # THERE IS NOTHING TO UPDATE!
        pass


class ReplayBufferTD:
    def __init__(self, buffer_size, input_dims):
        self.buffer_size = buffer_size
        self.input_dims = input_dims
        self.reset()

    def reset(self):
        self.counter = 0
        self.state_memory = np.zeros((self.buffer_size, *self.input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.buffer_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.buffer_size,dtype=np.int32)
        self.td_memory = np.zeros(self.buffer_size,dtype=np.float32)

    # sort all bugffers in descending order acc to td_memory
    # the highest td errors will then be at the lowest indices --> simple to extract
    def sort_td_wise(self):
      max_mem = min(self.counter, self.buffer_size)
      indices = np.argsort(self.td_memory[0:max_mem])[::-1] ;
      # descending sort
      self.state_memory[0:max_mem] = self.state_memory[0:max_mem][indices] ;
      self.new_state_memory[0:max_mem] = self.new_state_memory[0:max_mem][indices] ;
      self.action_memory[0:max_mem] = self.action_memory[0:max_mem][indices] ;
      self.reward_memory[0:max_mem] = self.reward_memory[0:max_mem][indices] ;
      self.terminal_memory[0:max_mem] = self.terminal_memory[0:max_mem][indices] ;
      self.td_memory[0:max_mem] = self.td_memory[0:max_mem][indices] ;

    def get_state_memory(self):
      max_mem = min(self.counter, self.buffer_size) ;
      if max_mem == 0:
        return None ;
      return self.state_memory[0:max_mem] ;


    def reset_buffer(self):
      self.counter = 0 ;


    def store_transition(self,state, action, reward, state_, done, td):
        index = self.counter % self.buffer_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.td_memory[index] = td ;
        self.terminal_memory[index] = 1 - int(done)
        self.counter += 1

    def getNrEntries(self):
      return min(self.counter, self.buffer_size) ;


    # if td_percentage is not None: assume buffer is td-sorted, draw percentage of highest TD error samples
    def sample_buffer(self, batch_size, td_percentage = None, replace = True):
        if td_percentage is None:
          max_mem = min(self.counter, self.buffer_size)
          batch = np.random.choice(max_mem, batch_size, replace=False)

          states = self.state_memory[batch]
          states_ = self.new_state_memory[batch]
          rewards = self.reward_memory[batch]
          actions = self.action_memory[batch]
          terminal = self.terminal_memory[batch]
        else:
          max_mem = min(self.counter, self.buffer_size)
          last_index = int(max_mem * td_percentage) ;
          print(max_mem, last_index, "--vs--", batch_size)
          #print("TD: maxmin=", self.td_memory.max(), self.td_memory.min(), "[0] and [10percent]=", self.td_memory[0], self.td_memory[last_index]) ;
          batch = np.random.choice(last_index, batch_size, replace=replace) ;
          states = self.state_memory[batch]
          states_ = self.new_state_memory[batch]
          rewards = self.reward_memory[batch]
          actions = self.action_memory[batch]
          terminal = self.terminal_memory[batch]
          

        return states, actions, rewards, states_, terminal, batch
    


class PrioritizedReplayBuffer:
    """ PER buffer that samples proportionately to the TD errors for each sample. """

    def __init__(self, buffer_size, input_dims, per_alpha=0.6, per_beta=1.0, per_eps=1e-6, per_delta_beta = 0.0001):
        """
        Args:
            buffer_size: max buffer size.
            input_dims: observation dimension.
            per_alpha: the strength of the prioritization (0.0 - no prioritization, 1.0 - full prioritization).
            per_beta: beta controls how much prioritization to apply, should start small (0,4-0,6 and anneal to 1).
            per_eps: small constant ensuring that each sample has some non-zero probability of being drawn.
        """
        self.buffer_size    = buffer_size
        self.alpha          = per_alpha
        self.beta0          = per_beta
        self.beta           = per_beta ;
        self.eps            = per_eps
        self.delta_beta     = per_delta_beta ;
        self.input_dims = input_dims
        self.reset()

    def reset(self):
        self.counter = 0

        self.state_memory       = np.zeros((self.buffer_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory   = np.zeros((self.buffer_size, *self.input_dims), dtype=np.float32)
        self.action_memory      = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory      = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_memory    = np.zeros(self.buffer_size, dtype=np.int32)
        self.priorities         = np.ones(self.buffer_size, dtype=np.float32)
        self.reset_beta()

    def store_transition(self, state, action, reward, state_, done):
        index = self.counter % self.buffer_size

        self.state_memory[index]        = state
        self.new_state_memory[index]    = state_
        self.reward_memory[index]       = reward
        self.action_memory[index]       = action
        self.terminal_memory[index]     = 1 - int(done)
        self.priorities[index]          = np.max(self.priorities)  # assign a priority, initially set to a high value.
        
        self.counter += 1
        #self.counter = min(self.counter, self.buffer_size)

    def calculate_probabilities(self):
        """ Calculates probability of being sampled for each element in the buffer.
        Returns:
            probabilities: 
                returns a probability distribution P(i) of how likely it is,
                that a buffer element will be retrieved according to its priority.
                We use the proportional variant here (see DeepMind paper).
        """
        priorities = self.priorities[:self.counter] ** self.alpha
        return priorities / sum(priorities[:self.counter])

    def calculate_importance(self, probs):
        """ Calculates the importance sampling bias correction. """
        #N = min(self.counter, self.buffer_size) ;
        importances = ( (1.0 / probs))**self.beta  
        return importances / np.max(importances)  # max w_i = 1 for stability

    def reset_beta(self):
      self.beta = self.beta0 ;
    

    def sample_buffer(self, batch_size):
        """ Sample based on priorities, experiences with a higher priority are more likely to be sampled. """
        # calculate probability distribution based on the importance (so samples with no/low importance still have a small chance to be sampled)
        probs = self.calculate_probabilities()
        # generate a probability distribution of size N (mini-batch size) based on the importance of already stored samples.
        possible_indices = np.arange(0, min(self.counter, self.buffer_size))
        # draw from distribution (samples with higher importance/td-error are drawn more frequently)
        batch_indices = np.random.choice(possible_indices, batch_size, p=probs[:self.counter])
        # calculate the importance weights for each drawn sample
        self.importance = self.calculate_importance(probs[batch_indices])

        states      = self.state_memory[batch_indices]
        states_     = self.new_state_memory[batch_indices]
        rewards     = self.reward_memory[batch_indices]
        actions     = self.action_memory[batch_indices]
        terminal    = self.terminal_memory[batch_indices]        

        ## anneal beta upwards, the conszant delta_beta must be chosen such that beta=1 at the end of a task
        self.beta += self.delta_beta ;

        return states, actions, rewards, states_, terminal, batch_indices

    def get_weights_current_batch(self):
      return self.importance ;

    def update_priorities(self, indices, td_errors):
        """Updates the priorities for a batch given the TD errors (pred. Q-values minus target Q-values).
        Experiences that are sampled often are given low priority values to balance sampling.
        Args:
            indices: np.array, the list of indices of the priority list to be updated.
            td_errors: np.array, the list of TD errors for those indices. The
                priorities will be updated to the TD errors plus the offset.
            eps: float, small positive value to ensure that all trajectories
                have some probability of being selected.
        """
        for index, error in zip(indices, td_errors):
            self.priorities[index] = abs(error) + self.eps
