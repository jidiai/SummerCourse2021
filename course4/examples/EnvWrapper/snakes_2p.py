import numpy as np
from EnvWrapper.BaseWrapper import BaseWrapper
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make
env = make("snakes_2p")


class snakes_2p(BaseWrapper):
    def __init__(self):
        self.env = env
        super().__init__(self.env)

    def get_actionspace(self):
        return self.env.get_action_dim()

    def get_observationspace(self):
        self.num_agents = self.env.n_player
        self.obs_dim = 2*self.num_agents + 14
        return self.obs_dim  # 18

    def step(self, action, train=True):
        '''
        return: next_state, reward, done, _, _
        '''

        next_state, reward, done, _, info = self.env.step(action)

        reward = np.array(reward)
        ctrl_agent_index = list(range(0, self.num_agents))  # [0,1]
        step_reward = get_reward(info, ctrl_agent_index, reward)
        next_state = [get_observations(next_state[id], id, obs_dim=self.obs_dim)
                      for id in range(self.env.n_player)]

        # obs_next = np.zeros((self.num_agents, self.get_observationspace))
        # for i, s_n in enumerate(next_state):
        #     obs_next_copy = s_n.copy()
        #     agent_id = obs_next_copy["controlled_snake_index"] - 2
        #     agents_index = [agent_id]
        #     obs_next[i] = get_observations(obs_next_copy, agents_index, obs_dim=self.get_observationspace)[:]

        return next_state, step_reward, done, _, _

    def reset(self):
        state = self.env.reset()
        state = [get_observations(state[id], id, obs_dim=self.obs_dim)
                      for id in range(self.env.n_player)]
        return state

    def close(self):
        pass

    def set_seed(self, seed):
        pass

    def make_render(self):
        self.env.env_core.render()


def get_reward(info, snake_index, reward):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] -= 10

    return step_reward


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


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
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, id, obs_dim):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((1, obs_dim)) # todo
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    agents_index = [id]
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions # todo: to check
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, element, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations.squeeze().tolist()




