import argparse ;

class ExplorationController(object):
    def __init__(self):
        self.exploration = True ;
        self.babbling = True ;
    
    def configure(self,curr_task,task):
      pass ;
    
    def query(self):
      return False ;

    def update_epsilon(self):
      pass ;

    def enable_exploration(self):
        self.exploration = True

    def disable_exploration(self):
        self.exploration = False

    def enable_babbling(self):
      self.babbling = True ;


    def disable_babbling(self):
      self.babbling = False ;

    def get_current_status(self):
        return self.babbling, self.exploration ;

    def define_base_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--seed',     type=int, default=42,                       help='The random seed for the experiment run.')

        parser.add_argument('--debug',                    type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        parser.add_argument('--exploration', type=str)
        parser.add_argument('--exploration_start_task',   type=int, default=0   ,     help='all taskl before this one are babbling')

    def parse_args(self):
        parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.', exit_on_error=False)
        self.define_base_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;

