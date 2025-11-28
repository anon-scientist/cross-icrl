import argparse
class BaseExploration:
    def __init__(self):
        self.base_config, _ = self.parse_base_args()
        self.name = self.base_config.exploration
        self.start_task = self.base_config.exploration_start_task

        self.current_step_in_episode = 0

        self.exploration = True ;
    
    def reset(self):
        raise Exception("NOT IMPLEMENTED EXCEPTION!")
    
    def configure(self):
        raise Exception("NOT IMPLEMENTED EXCEPTION!")
    
    def query(self):
        raise Exception("NOT IMPLEMENTED EXCEPTION!")

    def update(self):
        raise Exception("NOT IMPLEMENTED EXCEPTION!")

    def enable_exploration(self):
        self.exploration = True

    def disable_exploration(self):
        self.exploration = False

    def get_current_status(self):
        return ("NOSTATUS",)

    def define_base_args(self, parser):
        # ------------------------------------ LEARNER
        parser.add_argument('--seed',     type=int, default=42,                       help='The random seed for the experiment run.')
        parser.add_argument('--debug',                    type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        parser.add_argument('--exploration', type=str)
        parser.add_argument('--exploration_start_task',   type=int, default=0   ,     help='all taskl before this one are babbling')

    def parse_base_args(self):
        parser = argparse.ArgumentParser('BaseExploration', 'argparser for the BaseExplorationClass', exit_on_error=False)
        self.define_base_args(parser) ;
        config, unparsed = parser.parse_known_args()
        return config, unparsed ;