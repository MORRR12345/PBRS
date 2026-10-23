## Critic类

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, LR, state_dim, DEVICE):
        """价值函数网络"""
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),   #nn.LeakyReLU(),nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        ).to(DEVICE)
        self.optim =  torch.optim.Adam(self.parameters(), lr = LR)
    
    def forward(self, state):
        """前向传播"""
        dim = len(state.shape)
        if dim == 4:
            s_value = self.net(state.view(state.size(0), state.size(1), -1))  # 展平为 [step, N, n*state]
            return s_value.squeeze(-1)
        else:
            s_value = self.net(state.view(state.size(0), -1))  # 展平为 [N, n*state]
            return s_value.squeeze(-1)
