# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os

# load critic
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic


# TODO: Complete DQN algo under evaluation.
class DQN:
    def __init__(self):
        pass

    def choose_action(self, observation):
        pass

    def load(self, file):
        pass


#TODO: Complete a action wrapper. You can refer to hw1.
def action_from_algo_to_env(joint_action):
    pass


state_dim = 4
action_dim = 2
hidden_size = 64

# TODO: Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic.pth'
agent = DQN()
agent.load(critic_net)


# This function dont need to change.
def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['obs']
    action = agent.choose_action(obs)
    return action_from_algo_to_env(action)