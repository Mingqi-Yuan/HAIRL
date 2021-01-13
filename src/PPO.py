from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical

import numpy as np
import pandas as pd
import torch


class Actor(nn.Module):
    def __init__(self, kwargs):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(kwargs['input_dim'], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.leaky_relu(self.fc1(state))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        return x


class Critic(nn.Module):
    def __init__(self, kwargs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(kwargs['input_dim'], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, state):
        x = self.leaky_relu(self.fc1(state))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x


class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):  # 存储经验
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):  # 回放经验
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field \
                in self.memory.columns)


class PPOAgent():
    def __init__(self,
                 env,
                 device,
                 actor_kwargs,
                 critic_kwargs,
                 clip_ratio=0.1,
                 gamma=0.99,
                 lambd=0.99,
                 min_trajectory_length=100,
                 lr=1e-4,
                 batches=5,
                 batch_size=64):
        self.action_n = env.action_space.n
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lambd = lambd
        self.min_trajectory_length = min_trajectory_length
        self.lr = lr
        self.batches = batches
        self.batch_size = batch_size

        self.trajectory = []  # 存储回合内的轨迹
        self.replayer = PPOReplayer()

        self.device = device
        self.actor_net = Actor(actor_kwargs)
        self.critic_net = Critic(critic_kwargs)
        self.actor_net.to(device)
        self.critic_net.to(device)

        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), self.lr)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), self.lr)

    def ppo_loss(self, y_true, y_pred):  # 损失函数
        # 真实值 y_true : (2*action_n,) 旧策略的策略概率 + 优势函数
        # 预测值 y_pred : (action,) 神经网络输出的策略概率
        p = y_pred  # 新策略概率
        p_old = y_true[:, :self.action_n]  # 旧策略概率
        advantage = y_true[:, self.action_n:]  # 优势
        surrogate_advantage = (p / p_old) * advantage  # 代理优势
        clip_times_advantage = self.clip_ratio * advantage
        max_surrogate_advantage = advantage + torch.where(
            advantage > 0.,
            clip_times_advantage,
            - clip_times_advantage)
        clipped_surrogate_advantage = torch.minimum(surrogate_advantage, max_surrogate_advantage)

        loss = - torch.mean(clipped_surrogate_advantage, dim=[0, 1])

        return loss

    def learn(self, state, action, reward, done):
        self.trajectory.append((state, action, reward))
        if done:
            df = pd.DataFrame(self.trajectory, columns=['state', 'action', 'reward'])  # 开始对本回合经验进行重构

            states = torch.from_numpy(np.stack(df['state'])).float().to(self.device)
            df['v'] = self.critic_net(states).cpu().detach().numpy()
            pis = self.actor_net(states).cpu().detach().numpy()

            df['pi'] = [a.flatten() for a in np.split(pis, pis.shape[0])]
            df['next_v'] = df['v'].shift(-1).fillna(0.)
            df['u'] = df['reward'] + self.gamma * df['next_v']
            df['delta'] = df['u'] - df['v']  # 时序差分误差
            df['return'] = df['reward']  # 初始化优势估计，后续会再更新
            df['advantage'] = df['delta']  # 初始化优势估计，后续会再更新
            for i in df.index[-2::-1]:  # 指数加权平均
                df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
                df.loc[i, 'advantage'] += self.gamma * self.lambd * \
                                          df.loc[i + 1, 'advantage']  # 估计优势
            fields = ['state', 'action', 'pi', 'advantage', 'return']
            self.replayer.store(df[fields])  # 存储重构后的回合经验
            self.trajectory = []  # 为下一回合初始化回合内经验
            if len(self.replayer.memory) > self.min_trajectory_length:
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                for batch in range(self.batches):
                    states, actions, pis, advantages, returns = self.replayer.sample(size=self.batch_size)
                    ext_advantages = np.zeros_like(pis)
                    ext_advantages[range(self.batch_size), actions] = advantages

                    states = torch.from_numpy(states).float().to(self.device)
                    actor_targets = torch.from_numpy(np.hstack([pis, ext_advantages])).float().to(self.device)  # 执行者目标
                    returns = torch.from_numpy(returns).float().to(self.device)

                    actor_X = self.actor_net(states)
                    critic_X = self.critic_net(states)

                    actor_loss = self.ppo_loss(actor_targets, actor_X)
                    actor_loss.backward()
                    critic_loss = F.mse_loss(critic_X.squeeze(1), returns)
                    critic_loss.backward()
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()

                self.replayer = PPOReplayer()  # 为下一回合初始化经验回放

    def decide(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        probs = self.actor_net(state)

        # action = torch.argmax(probs, dim=1)
        dist = Categorical(probs)
        action = dist.sample()

        return action.cpu().detach().numpy()[0]

    def save(self, save_dir):
        torch.save(self.actor_net, save_dir)

    def update_policy(self, df):
        states = torch.from_numpy(np.stack(df['state'])).float().to(self.device)
        df['v'] = self.critic_net(states).cpu().detach().numpy()
        pis = self.actor_net(states).cpu().detach().numpy()

        df['pi'] = [a.flatten() for a in np.split(pis, pis.shape[0])]
        df['next_v'] = df['v'].shift(-1).fillna(0.)
        df['u'] = df['reward'] + self.gamma * df['next_v']
        df['delta'] = df['u'] - df['v']  # 时序差分误差
        df['return'] = df['reward']  # 初始化优势估计，后续会再更新
        df['advantage'] = df['delta']  # 初始化优势估计，后续会再更新
        for i in df.index[-2::-1]:  # 指数加权平均
            df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
            df.loc[i, 'advantage'] += self.gamma * self.lambd * \
                                      df.loc[i + 1, 'advantage']  # 估计优势
        fields = ['state', 'action', 'pi', 'advantage', 'return']
        self.replayer.store(df[fields])  # 存储重构后的回合经验

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        for batch in range(self.batches):
            states, actions, pis, advantages, returns = self.replayer.sample(size=self.batch_size)
            ext_advantages = np.zeros_like(pis)
            ext_advantages[range(self.batch_size), actions] = advantages

            states = torch.from_numpy(states).float().to(self.device)
            actor_targets = torch.from_numpy(np.hstack([pis, ext_advantages])).float().to(self.device)  # 执行者目标
            returns = torch.from_numpy(returns).float().to(self.device)

            actor_X = self.actor_net(states)
            critic_X = self.critic_net(states)

            actor_loss = self.ppo_loss(actor_targets, actor_X)
            actor_loss.backward()
            critic_loss = F.mse_loss(critic_X.squeeze(1), returns)
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.replayer = PPOReplayer()  # 为下一回合初始化经验回放