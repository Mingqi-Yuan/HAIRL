from torch import nn, optim
from torch.nn import functional as F
from scipy import stats
import numpy as np

from src.Networks import *
from src.PPO import *
from src.data import *

class ExtrinsicRewardBlock:
    def __init__(
            self,
            env,
            device,
            writer,
            expert_agent,
            dis_kwargs,
            lr,
            batch_size
    ):
        self.env = env
        self.device = device
        self.writer = writer
        self.lr = lr
        self.batch_size = batch_size
        
        self.expert_agent = expert_agent 
        self.dis = Discriminator(kwargs=dis_kwargs)
        self.dis.to(self.device)
        self.optimizer_dis = optim.RMSprop(self.dis.parameters(), self.lr)

    def update_erb(self, traj_g_loader, epoch):
        batches = len(traj_g_loader)
        total_dis_loss = 0.
        total_JS = 0.
        for idx, g_data in enumerate(traj_g_loader):
            self.optimizer_dis.zero_grad()

            g_states, g_actions, g_rewards, g_next_states = g_data
            g_actions = F.one_hot(g_actions, self.env.action_space.n).float()
            ''' get expert trajs '''
            e_actions = self.expert_agent(g_states.to(self.device))
            e_actions = torch.argmax(e_actions, dim=1)
            e_actions = F.one_hot(e_actions, self.env.action_space.n).float()

            D_fake = self.dis(g_states.to(self.device), g_actions.to(self.device))
            D_real = self.dis(g_states.to(self.device), e_actions.to(self.device))
            dis_loss = self.wasserstein_loss(D_fake, D_real)
            dis_loss.backward()
            self.optimizer_dis.step()

            total_dis_loss += dis_loss.item()
            total_JS += self.get_JS(g_actions, e_actions)

        self.writer.add_scalar('ERB/DISCRIMINATOR LOSS', total_dis_loss / batches, global_step=epoch)
        self.writer.add_scalar('ERB/JS DIVERGENCE', total_JS / batches, global_step=epoch)

        return total_dis_loss / batches, total_JS / batches

    def get_JS(self, g_actions, e_actions):
        g_actions = np.argmax(g_actions.cpu().numpy(), axis=1) + 1
        e_actions = np.argmax(e_actions.cpu().numpy(), axis=1) + 1
        JS = 0.5 * stats.entropy(g_actions, e_actions) + 0.5 * stats.entropy(e_actions, g_actions)

        return JS

    def wasserstein_loss(self, D_fake, D_real):
        return torch.mean(D_fake) - torch.mean(D_real)

    def cal_extrinsic_reward(self, g_states, g_actions):
        g_actions = F.one_hot(g_actions, self.env.action_space.n).float()
        extrinsic_rewards = self.dis(g_states.to(self.device), g_actions.to(self.device))

        return extrinsic_rewards.squeeze(1).detach().cpu()

class IntrinsicRewardBlock:
    def __init__(
            self,
            env,
            device,
            writer,
            inverse_kwargs,
            forward_kwargs,
            lr,
            batch_size
    ):
        self.env = env
        self.device = device
        self.writer = writer
        self.lr = lr
        self.batch_size = batch_size

        self.im = InverseModel(kwargs=inverse_kwargs)
        self.fm = ForwardModel(kwargs=forward_kwargs)
        self.im.to(self.device)
        self.fm.to(self.device)
        self.optimizer_im = optim.RMSprop(self.im.parameters(), self.lr)
        self.optimizer_fm = optim.RMSprop(self.fm.parameters(), self.lr)

    def update_irb(self, traj_g_loader, epoch):
        batches = len(traj_g_loader)
        total_im_loss = 0.
        total_fm_loss = 0.
        for idx, g_data in enumerate(traj_g_loader):
            self.optimizer_im.zero_grad()
            self.optimizer_fm.zero_grad()

            g_states, g_actions, g_rewards, g_next_states = g_data

            pred_actions = self.im(g_states.to(self.device), g_next_states.to(self.device))
            pred_next_states = self.fm(g_states.to(self.device), pred_actions)

            im_loss = F.cross_entropy(pred_actions, g_actions.to(self.device))
            fm_loss = 0.5 * F.mse_loss(pred_next_states, g_next_states.to(self.device))
            im_loss.backward(retain_graph=True)
            fm_loss.backward(retain_graph=True)
            self.optimizer_im.step()
            self.optimizer_fm.step()

            total_im_loss += im_loss.item()
            total_fm_loss += fm_loss.item()

        self.writer.add_scalar('IRB/INVERSE MODEL LOSS', total_im_loss / batches, global_step=epoch)
        self.writer.add_scalar('IRB/FORWARD MODEL LOSS', total_fm_loss / batches, global_step=epoch)

        return total_im_loss / batches, total_fm_loss / batches

    def cal_intrinsic_reward(self, g_states, g_next_states):
        pred_actions = self.im(g_states.to(self.device), g_next_states.to(self.device))
        pred_next_states = self.fm(g_states.to(self.device), pred_actions)

        intrinsic_rewards = torch.square(g_next_states.to(self.device) - pred_next_states)
        intrinsic_rewards = torch.sum(intrinsic_rewards, dim=1)

        return intrinsic_rewards.detach().cpu()
            
class HAIRL:
    def __init__(
            self,
            env,
            device,
            writer,
            actor_kwargs,
            critic_kwargs,
            dis_kwargs,
            inverse_kwargs,
            forward_kwargs,
            expert_agent,
            lr,
            batch_size,
            shuffle
    ):
        self.env = env
        self.device = device
        self.writer = writer
        self.lr = lr
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.agent = PPOAgent(
            env=self.env,
            device=self.device,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs
        )

        self.expert_agent = expert_agent
        self.erb = ExtrinsicRewardBlock(
            env=self.env,
            device=self.device,
            writer=self.writer,
            expert_agent=self.expert_agent,
            dis_kwargs=dis_kwargs,
            lr=self.lr,
            batch_size=self.batch_size
        )

        self.irb = IntrinsicRewardBlock(
            env=self.env,
            device=self.device,
            writer=self.writer,
            inverse_kwargs=inverse_kwargs,
            forward_kwargs=forward_kwargs,
            lr=self.lr,
            batch_size=self.batch_size
        )

    def execute_policy(self):
        traj_g = list()
        episode_reward = 0

        state = self.env.reset()
        while True:
            action = self.agent.decide(state)
            next_state, reward, done, info = self.env.step(action)
            traj_g.append((state, action, reward))
            episode_reward += reward

            if done:
                break

            state = next_state

        return traj_g, len(traj_g), episode_reward

    def cal_hybrid_reward(self, df, ratio):
        df_length = len(df)
        g_states_arr = np.stack(df['state'][:df_length - 1]).astype('float32')

        g_states = torch.from_numpy(g_states_arr)
        g_actions = torch.from_numpy(np.stack(df['action'][:df_length - 1]))
        g_next_states = torch.from_numpy(g_states_arr[:df_length])

        er = self.erb.cal_extrinsic_reward(g_states, g_actions)
        ir = self.irb.cal_intrinsic_reward(g_states, g_next_states)

        hybrid_rewards = ratio * er + (1.0 - ratio) * ir

        return hybrid_rewards.numpy()

    def cal_ratio(self, epoch):
        ratio = np.exp(- epoch / 50)

        self.writer.add_scalar('RATIO', ratio, global_step=epoch)

        return ratio

    def train(self, epoch):
        traj_g, traj_g_length, episode_reward = self.execute_policy()
        traj_g_df = pd.DataFrame(traj_g, columns=['state', 'action', 'reward'])
        traj_g_loader = build_loader(traj_g_df, traj_g_length, self.batch_size, self.shuffle)

        ''' update extrinsic reward block '''
        erb_dis_loss, erb_JS = self.erb.update_erb(traj_g_loader, epoch)
        ''' update intrinsic reward block '''
        irb_im_loss, irb_fm_loss = self.irb.update_irb(traj_g_loader, epoch)

        ''' calculate hybrid reward '''
        ratio = self.cal_ratio(epoch)
        hybrid_rewards = self.cal_hybrid_reward(traj_g_df, ratio)
        traj_g_df['reward'][:traj_g_length - 1] = hybrid_rewards
        traj_g_df.drop(traj_g_df.index[traj_g_length - 1])

        ''' update policy '''
        self.agent.update_policy(traj_g_df)

        print(
            'INFO: Epoch={}, '
            'ERB DIS LOSS={:.5f}, '
            'ERB JS={:.5f}, '
            'IRB IM LOSS={:.5f}, '
            'IRB FM LOSS={:.5f}, '
            'EPISODE REWARD={:.5f}'.format(
                epoch + 1,
                erb_dis_loss,
                erb_JS,
                irb_im_loss,
                irb_fm_loss,
                episode_reward
            ))

        ''' save model '''
        if (epoch + 1) % 10:
            self.agent.save(save_dir='./models/hairl/hairl_epochs{}.pth'.format(epoch+1))



