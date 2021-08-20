import torch.nn as nn

# ====================================== helper functions ======================================
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from common import make_grid_map, get_surrounding, get_observations


# ====================================== define algo ===========================================
# todo
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        pass


# todo
class DQN(object):
    def __init__(self):
        pass
    def load(self, file):
        pass


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
#todo
agent = DQN()
agent.load('critic_net.pth')


# ================================================================================================
"""
input:
    observation: dict
    {
        1: 豆子，
        2: 第一条蛇的位置，
        3：第二条蛇的位置，
        "board_width": 地图的宽，
        "board_height"：地图的高，
        "last_direction"：上一步各个蛇的方向，
        "controlled_snake_index"：当前你控制的蛇的序号（2或3）
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""
# todo
def my_controller(observation, action_space_list, is_act_continuous):
    pass
