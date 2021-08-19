import importlib
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from agents.baseagent import Baseagent
from networks.critic import Critic

def ini_agents(args):
    agent_file_name = str("algo." + str(args.algo) + "." + str(args.algo))
    agent_file_import = importlib.import_module(agent_file_name)
    agent_class_name = args.algo.upper()

    # 实例化agent
    agent = getattr(agent_file_import, agent_class_name)(args, )
    return agent


class MultiRLAgents(Baseagent):
    def __init__(self, args):
        super(MultiRLAgents, self).__init__(args)
        self.args = args
        self.agents = list()
        self.given_net = Critic(self.args.obs_space,  self.args.action_space, self.args.hidden_size)
        for i in range(self.args.n_player):
            agent_file_name = str("algo." + str(self.args.algo) + "." + str(self.args.algo))
            agent_file_import = importlib.import_module(agent_file_name)
            agent_class_name = self.args.algo.upper()
            if self.args.share_net:
                given_net = self.given_net
            else:
                given_net = Critic(self.args.obs_space,  self.args.action_space, self.args.hidden_size)
            agent = getattr(agent_file_import, agent_class_name)(args, given_net)
            self.agents.append(agent)

    def action_from_algo_to_env(self, joint_action):
        '''
        :param joint_action:
        :return: wrapped joint action: one-hot
        '''
        joint_action_ = []
        for a in range(1):
            action_a = joint_action["action"]
            if not self.args.action_continuous:  # discrete action space
                each = [0] * self.args.action_space
                each[action_a] = 1
                joint_action_.append(each)
            else:
                joint_action_.append(action_a)  # continuous action space
        return joint_action_

    # ======================================= inference =============================================
    def choose_action_to_env(self, observation, id, train=True):
        obs_copy = observation.copy()
        action_from_algo = self.agents[id].choose_action(obs_copy, train)
        action_to_env = self.action_from_algo_to_env(action_from_algo)
        return action_to_env

    # ====================================== update algo =============================================
    def learn(self):
        for agent in self.agents:
            agent.learn()

    def save(self, save_path, episode):
        for id, agent in enumerate(self.agents):
            agent.save(save_path, episode, id)
