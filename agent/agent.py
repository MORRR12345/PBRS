## 智能体类（Agent） 包含了 actor、critic、messer 以及相关更新函数

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agent.actor import Actor
from agent.critic import Critic
from tool import find_path

#################################### 学习参数 ####################################

ACTOR_LR = 1e-4          # actor网络学习率，常用范围1e-4
CRITIC_LR = 1e-3         # critic网络学习率，常用范围1e-3

GAMMA = 0.99             # 折扣率
PPO_CLIP = 0.2           # PPO裁剪参数
GAE_LAMBDA = 0.95        # GAE参数

VALUE_COEF = 0.5         # 价值损失系数
ENTROPY_COEF = 0.01      # 熵正则化系数

AGENT_TRAIN_BATCH = 256  # 更新使用批量大小

####################################    agent 类    ####################################

class Agent():
    """智能体类"""

    def __init__(self, state_dim, act_dim, DEVICE):
        self.state_dim = state_dim
        self.act_dim = act_dim

        #* actor网络模块
        self.actor = Actor(ACTOR_LR, state_dim, act_dim, DEVICE)

        #* critic网络模块
        self.critic = Critic(CRITIC_LR, state_dim, DEVICE)

    def load(self, type, time, epoch):
        """加载模型"""
        path = find_path(time, epoch, type) 
        dict = torch.load(f"{path}", weights_only=True)
        if type == "actor":
            self.actor.load_state_dict(dict)
        elif type == "critic":
            self.critic.load_state_dict(dict)
    
    def get_action(self, observations):
        """获取所有智能体动作"""
        dist, mu = self.actor(observations) # [N,obs] -> [N,action]
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(-1) # [N,action] -> [N]
        return actions, mu, log_probs
    
    #*################################### actor和critic 更新函数 ####################################

    def compute_gae(self, rewards, values, next_values, dones, active=None):
        """计算广义优势函数GAE"""
        with torch.no_grad():
            steps = len(rewards) # 步数steps
            num = rewards.size(1)  # 环境数量N
            non_terminal = (~dones.bool()).to(rewards.dtype)
            deltas = rewards + GAMMA * next_values * non_terminal - values

            # 初始化
            advantages = torch.zeros(steps, num, device=rewards.device) # [steps, N]
            gae = torch.zeros(num, device=rewards.device) # [N]

            # 从后往前计算 GAE
            for t in reversed(range(steps)):
                gae = deltas[t] + GAMMA * GAE_LAMBDA * non_terminal[t] * gae
                advantages[t] = gae
            returns = advantages + values
            # 只用有效轨迹归一化，排除其他环境结束后的填充步。
            if active is None:
                active = torch.ones_like(dones, dtype=torch.bool)
            valid_advantages = advantages[active.bool()]
            mean = valid_advantages.mean()
            std = valid_advantages.std(unbiased=False)
            advantages = (advantages - mean) / (std + 1e-8)
        return advantages, returns

    # 广义优势函数效果更好，但是可能需要随环境设计而改变，下面这个更通用
    # def compute_gae(self, rewards, values, next_values): 
    #     """计算普通优势函数"""
    #     with torch.no_grad():
    #         advantages = rewards + GAMMA * next_values - values  # [steps, N]
    #         returns = advantages + values
    #         # 归一化优势
    #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #     return advantages, returns
    
    def update_actor_critic(self, expe_data):
        """更新actor和critic""" 
        obs = expe_data['obs'].detach()                 # [steps,N,n,obs]
        actions = expe_data['actions'].detach()         # [steps,N,n,action]
        oldprobs = expe_data['log_probs'].detach()      # [steps,N]
        rewards = expe_data['rewards'].detach()         # [steps,N]
        nextobs = expe_data['next_obs'].detach()        # [steps,N,n,obs]
        dones = expe_data['dones'].detach()              # [steps,N]
        active = expe_data['active'].detach().bool()     # [steps,N]

        actor_losses = []
        critic_losses = []

        with torch.no_grad():
            values = self.critic(obs)   # [steps,N,n,obs] -> [steps,N]
            nextvalues = self.critic(nextobs)   # [steps,N,n,obs] -> [steps,N]
            advantages, returns = self.compute_gae(
                rewards, values, nextvalues, dones, active
            )

        # 创建数据集
        flat_active = active.view(-1)
        dataset = TensorDataset(
            obs.view(-1, obs.size(2), obs.size(3))[flat_active],
            actions.view(-1, actions.size(2), actions.size(3))[flat_active],
            oldprobs.view(-1)[flat_active],
            advantages.view(-1)[flat_active],
            returns.view(-1)[flat_active],
        )

        # 多次迭代更新
        for _ in range(4):

            # shuffle=True乱批次顺序，drop_last=True放弃最后一个小批次
            loader = DataLoader(
                dataset,
                batch_size=min(AGENT_TRAIN_BATCH, len(dataset)),
                shuffle=True,
                drop_last=False,
            )

            for batch in loader:
                #* 加载信息
                obs_b, actions_b, oldprobs_b, advantages_b, returns_b = batch

                #* Critic损失
                values = self.critic(obs_b)   # [batch,state] -> [batch]
                critic_loss = nn.functional.mse_loss(values, returns_b)

                #* Actor损失
                # 计算比例
                dist, _ = self.actor(obs_b) # [batch,state] -> [batch,action]
                newprobs = dist.log_prob(actions_b).sum(-1) # [batch,action] -> [batch]
                ratio = torch.exp(newprobs - oldprobs_b)  # [batch] -> [batch]

                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0-PPO_CLIP, 1.0+PPO_CLIP) * advantages_b
                actor_loss = - torch.min(surr1, surr2).mean()

                #* 熵损失
                entropy_loss = dist.entropy().mean()

                #* 总损失
                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_loss

                #* 更新Actor和Critic
                self.actor.optim.zero_grad()
                self.critic.optim.zero_grad()
                loss.backward()
                self.actor.optim.step()
                self.critic.optim.step()

                # 记录损失
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        actor_loss_mean = torch.stack(actor_losses).mean()
        critic_loss_mean = torch.stack(critic_losses).mean()
        return actor_loss_mean.item(), critic_loss_mean.item()
