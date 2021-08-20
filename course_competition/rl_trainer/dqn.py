import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DQN(object):
    def __init__(self, state_dim, action_dim, num_agent, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agent = num_agent

        self.hidden_size = args.hidden_size
        self.lr = args.lr_c
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.output_activation = args.output_activation

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.buffer = []
        self.loss = None

        # epsilon
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        self.learn_step_counter = 0
        self.target_replace_iter = args.target_replace

    def choose_action(self, observation, train=True):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        if train:
            self.eps = max(self.eps_end, self.eps - self.eps_delay)
            if random.random() < self.eps:
                action = random.randrange(self.action_dim)
            else:
                action = torch.argmax(self.critic_eval(observation)).item()
                # action = self.critic_eval(observation)
        else:
            action = torch.argmax(self.critic_eval(observation)).item()
            # action = self.critic_eval(observation)
        return action

    def store_transition(self, transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        obs, action, reward, obs_, done = zip(*samples)

        obs = torch.tensor(obs, dtype=torch.float).squeeze()
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(self.batch_size, -1).squeeze()
        obs_ = torch.tensor(obs_, dtype=torch.float).squeeze()
        done = torch.tensor(done, dtype=torch.float).view(self.batch_size, -1).squeeze()

        q_eval = self.critic_eval(obs).gather(1, action)
        q_next = self.critic_target(obs_).detach()
        q_target = (reward + self.gamma * q_next.max(1)[0] * (1 - done)).view(self.batch_size, 1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.learn_step_counter = 0
            self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.learn_step_counter += 1

        self.loss = loss.item()

        return loss

    def save(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_eval.state_dict(), model_critic_path)

    def load(self, file):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file))
        self.critic_target.load_state_dict(torch.load(file))
