import numpy as np
import torch
import torch.nn as nn
from typing import Union
from torch.distributions import Categorical
from types import SimpleNamespace as SN
import yaml
import os
import copy
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tgt_param.data * (1.0 - tau) + src_param.data * tau)


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agent_trained_index, obs_dim):
    state_copy = state.copy()

    agents_index = state_copy["controlled_snake_index"]

    if agents_index != agent_trained_index:
        error = "训练的智能体：{name}, 观测的智能体：{url}".format(name=agents_index, url=agent_trained_index)
        raise Exception(error)

    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    snakes_position = np.array(snakes_positions[agents_index], dtype=object)

    beans_position = np.array(beans_positions).flatten()

    observations = np.zeros((1, obs_dim)) # todo

    # self head position
    observations[0][:2] = snakes_position[0][:]

    # head surroundings
    head_x = snakes_position[0][1]
    head_y = snakes_position[0][0]
    head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
    observations[0][2:6] = head_surrounding[:]

    # beans positions
    observations[0][6:16] = beans_position[:]

    # other snake head positions
    snakes_other_position = np.array(snakes_positions[3], dtype=object) # todo
    observations[0][16:] = snakes_other_position[0][:]

    return observations


def get_reward(state, snake_index, reward, final_result):

    state_copy = state.copy()
    agents_index = state_copy["controlled_snake_index"]
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3}}
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)

    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if final_result == 1:       # done and won
            step_reward[i] += 100
        elif final_result == 2:     # done and lose
            step_reward[i] -= 50
        elif final_result == 3:     # done and draw
            step_reward[i] -= 20
        else:                       # not done
            if reward[i] > 0:           # eat a bean
                step_reward[i] += 20
            else:                       # just move
                snakes_position = np.array(snakes_positions[agents_index], dtype=object)
                beans_position = np.array(beans_positions, dtype=object)
                snake_heads = [snake[0] for snake in snakes_position]
                self_head = np.array(snake_heads[i])
                dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                step_reward[i] -= min(dists)
                if reward[i] < 0:
                    step_reward[i] -= 10
    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def append_random(act_dim, action):
    action = torch.Tensor([action]).to(device)
    acs = [out for out in action]
    num_agents = len(action)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def logits_greedy(state, info, logits, height, width):
    state = np.squeeze(np.array(state), axis=2)
    beans = info['beans_position']
    snakes = info['snakes_position']

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])
    greedy_action = greedy_snake(state, beans, snakes, width, height, [1])

    action_list = np.zeros(2)
    action_list[0] = logits_action[0]
    action_list[1] = greedy_action[0]

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


def load_config(args, log_path):
    file = open(os.path.join(str(log_path), 'config.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    args = SN(**config_dict)
    return args


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


