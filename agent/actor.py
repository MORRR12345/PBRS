## Actor类

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, LR, obs_dim, action_dim, DEVICE):
        """决策生成器，根据局部消息做出动作"""
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
            nn.Softsign()  # =X/(1+|X|) # 输出范围在[-1, 1]之间
        ).to(DEVICE)
        self.log_sigma = nn.Parameter(torch.zeros(action_dim)).to(DEVICE) # 直接设置为可学习的参数
        self.optim =  torch.optim.Adam(self.parameters(), lr = LR)

    def forward(self, obs):
        mu = self.net(obs)  
        std = torch.exp(self.log_sigma) 
        dist = torch.distributions.Normal(mu, std) 
        return dist, mu
