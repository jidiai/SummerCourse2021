# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/8/6 2:49 下午
# 描述：将运行的脚本记录在games/logs/下； 根据input_action_type类型，选择不同的动作获取函数。保存的log可以使用render_from_log回放。
import os.path
import os
import time
from myagent import my_agent
from randomagent import get_random_joint_act
import json
from env.chooseenv import make
from utils.get_logger import get_logger
import numpy as np
from copy import deepcopy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

# 0:命令行输入 1：random-agent 2:评测算法
input_action_type = 2


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_actions(g, joint_action_space, player_ids, info_before=''):
    if input_action_type == 0:
        return g.get_terminal_actions()

    if input_action_type == 1:
        return get_random_joint_act(joint_action_space)

    if input_action_type == 2:
        return get_joint_action_from_observation(g, player_ids, info_before)


def get_joint_action_from_observation(env, multi_part_agent_ids, all_observes):
    joint_action = []
    for policy_i in range(len(env.agent_nums)):
        agent_id_list = multi_part_agent_ids[policy_i]
        for agent_id in agent_id_list:
            a_obs = deepcopy(all_observes[agent_id])
            a_action = my_agent(a_obs, env.get_single_action_space(agent_id), None, env.is_act_continuous)
            joint_action.append(a_action)

    return joint_action


def get_joint_action_eval(game, multi_part_agent_ids, file_list, actions_spaces, t_agents_id, all_observes):
    if len(file_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(file_list), len(game.agent_nums))
        raise Exception(error)

    joint_action = []
    for policy_i in range(len(file_list)):
        agents_id_list = multi_part_agent_ids[policy_i]
        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = deepcopy(all_observes[agent_id])
            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


def run_game_local(g, player_ids):
    """
    This function is used to generate log for pygame rendering locally. Only saves .log file
    """
    log_path = os.getcwd() + '/logs/'
    logger = get_logger(log_path, g.game_name, save_file=True, json_file=False)

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_type, "n_player": g.n_player, "board_height": g.board_height,
                 "board_width": g.board_width, "init_info": str(g.init_info),
                 "init_state": str(g.get_render_data(g.current_state)),
                 "start_time": st,
                 "mode": "terminal",
                 "render_info": {"color": g.colors, "grid_unit": g.grid_unit, "fix": g.grid_unit_fix}}

    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        game_info[step] = {}
        game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        joint_act = get_actions(g, g.joint_action_space, player_ids)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            game_info[step]["info_before"] = info_before
        game_info[step]["joint_action"] = str(joint_act)
        game_info[step]["state"] = str(g.get_render_data(g.current_state))
        game_info[step]["reward"] = str(reward)

        if info_after:
            game_info[step]["info_after"] = info_after
        print("--------------------------------------------------------")

    game_info["winner"] = g.check_win()
    game_info["winner_information"] = str(g.won)
    game_info["n_return"] = str(g.n_return)
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    json_object = json.dumps(game_info, indent=4, ensure_ascii=False)
    logger.info(json_object)


def run_game(g, env_name, multi_part_agent_ids=None, is_eval=False, t_path_list=None, file_list=None,
             actions_spaces=None, agents_id=None, save_file=True, json_file=True):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    log_path = os.getcwd() + '/logs/'
    logger = get_logger(log_path, g.game_name, save_file, json_file)
    set_seed(g, env_name)

    if is_eval:
        for i in range(len(t_path_list)):
            file_path = t_path_list[i]

            import_path = '.'.join(file_path.split('/')[-5:])[:-3]
            function_name = 'm%d' % i
            import_name = "my_controller"
            import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
            exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    if g.is_obs_continuous:
        game_info = {"game_name": env_name, "n_player": g.n_player, "init_info": g.init_info,
                     # "init_state": g.get_render_data(g.current_state),
                     "start_time": st,
                     "mode": "terminal", "render_info": None,
                     "seed": g.seed if hasattr(g, "seed") else None,
                     "map_size": g.map_size if hasattr(g, "map_size") else None}
    else:
        if is_eval:
            render_info = None
        else:
            render_info = {"color": g.colors if hasattr(g, "color") else None,
                           "grid_unit": g.grid_unit if hasattr(g, "grid_unit") else None,
                           "fix": g.grid_unit_fix if hasattr(g, "grid_unit_fix") else None}

        game_info = {"game_name": env_name, "n_player": g.n_player,
                     "board_height": g.board_height if hasattr(g, "board_height") else None,
                     "board_width": g.board_width if hasattr(g, "board_width") else None,
                     "init_info": g.init_info,
                     # "init_state": g.get_render_data(g.current_state),
                     "start_time": st,
                     "mode": "terminal", "render_info": render_info,
                     "seed": g.seed if hasattr(g, "seed") else None,
                     "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes
    while not g.is_terminal():
        step = "step%d" % g.step_cnt
        if g.step_cnt % 10 == 0:
            print(step)
        info_dict = {}
        info_dict["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        if not is_eval:
            joint_act = get_actions(g, g.joint_action_space, multi_part_agent_ids, all_observes)
        else:
            joint_act = get_joint_action_eval(g, multi_part_agent_ids, file_list, actions_spaces, agents_id, all_observes)
        all_observes, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            info_dict["info_before"] = info_before
        if env_name.split("-")[0] in ["magent"]:
            info_dict["joint_action"] = g.decode(joint_act)
        info_dict["reward"] = reward

        if info_after:
            info_dict["info_after"] = info_after
        steps.append(info_dict)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)

    if not is_eval:
        logger.info(logs)

    results = {}
    if is_eval:
        results = get_results(g.n_return, g, agents_id)
    return results, logs


# 将 n_return 转化成 dict
def get_results(n_return, g, agents_id):
    if len(n_return) != sum(g.agent_nums):
        error = "return %d与 agent %d维度不同" % (len(n_return), sum(g.agent_nums))
        raise Exception(error)

    re = {}
    k = 0
    for i in range(len(agents_id)):
        model = str(agents_id[i])
        re[model] = {}
        sum_return = 0
        for j in range(g.agent_nums[i]):
            re[model][j + 1] = n_return[k]
            sum_return += n_return[k]
            k += 1
        re[model]["sum_return"] = sum_return
        re[model] = json.dumps(re[model])
    return re


def render_game(g, fps=1):
    """
    This function is used to generate log for pygame rendering locally and render in time. Only saves .log file
    """

    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)
    clock = pygame.time.Clock()
    log_path = os.getcwd() + '/logs/'
    logger = get_logger(log_path, env_type, save_file=True, json_file=False)

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = dict(game_name=env_type, n_player=g.n_player, board_height=g.board_height, board_width=g.board_width,
                     init_state=str(g.get_render_data(g.current_state)), init_info=str(g.init_info), start_time=st,
                     mode="window", render_info={"color": g.colors, "grid_unit": g.grid_unit, "fix": g.grid_unit_fix})

    all_observes = g.all_observes
    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        step = "step%d" % g.step_cnt
        print(step)
        game_info[step] = {}
        game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        joint_act = get_actions(g, g.joint_action_space, multi_part_agent_ids, all_observes)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            game_info[step]["info_before"] = info_before
        game_info[step]["joint_action"] = str(joint_act)

        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()

        game_info[step]["state"] = str(g.get_render_data(g.current_state))
        game_info[step]["reward"] = str(reward)

        if info_after:
            game_info[step]["info_after"] = info_after

        clock.tick(fps)

    game_info["winner"] = g.check_win()
    game_info["winner_information"] = str(g.won)
    game_info["n_return"] = str(g.n_return)
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    # json_object = json.dumps(game_info, indent=4, ensure_ascii=False)
    # logger.info(json_object)


if __name__ == "__main__":
    # "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "sokoban_1p", "snakes_3v3", "snakes_5p", "transport_2p",
    # "seek_2p", "classic_CartPole-v0", "classic_MountainCar-v0", "classic_MountainCarContinuous-v0",
    # "classic_Pendulum-v0", "classic_Acrobot-v1", "football_11v11_kaggle",
    # "MiniWorld-Hallway-v0", "MiniWorld-OneRoom-v0", "MiniWorld-OneRoomS6-v0", "MiniWorld-OneRoomS6Fast-v0",
    # "MiniWorld-TMaze-v0", "MiniWorld-TMazeLeft-v0", "MiniWorld-TMazeRight-v0", "MiniGrid-DoorKey-16x16-v0",
    # "MiniGrid-MultiRoom-N6-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0", "ParticleEnv-simple",
    # "ParticleEnv-simple_adversary", "ParticleEnv-simple_crypto", "ParticleEnv-simple_push",
    # "ParticleEnv-simple_reference", "ParticleEnv-simple_speaker_listener", "ParticleEnv-simple_spread",
    # "ParticleEnv-simple_tag", "ParticleEnv-simple_world_comm", "football_11_vs_11_stochastic",
    # "overcookedai-cramped_room", "overcookedai-asymmetric_advantages", "overcookedai-coordination_ring",
    # "overcookedai-forced_coordination", "overcookedai-counter_circuit", "magent-battle_v3-12v12",
    # "magent-battle_v3-20v20", "gridworld", "cliffwalking"
    env_type = "cliffwalking"
    render_mode = True
    run_local = False

    game = make(env_type, seed=None, conf=None)
    # game.reset()
    # 当前只支持myagent中的策略进行self play
    # policy_list = ["myagent"] or ["myagent","myagent"]

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    if render_mode:
        render_game(game)
    elif run_local:
        run_game_local(game, multi_part_agent_ids)
    else:
        _, _ = run_game(game, env_type, multi_part_agent_ids, is_eval=False)