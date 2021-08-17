# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os

# todo
# Once start to train, u can get saved model. Here we just say it is q_table.pth.
q_table = os.path.dirname(os.path.abspath(__file__)) + '/q_table.pth'
q_values = np.loadtxt(q_table, delimiter=",")


# todo
def action_from_algo_to_env(joint_action):
    pass

# todo
def behaviour_policy(q):
    pass


# todo
def epsilon_greedy(q_values):
    pass


# todo
def my_controller(observation, action_space, is_act_continuous=False):
    pass