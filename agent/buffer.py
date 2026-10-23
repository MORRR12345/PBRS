## 经验池

import torch
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################    经验池    ####################################
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 核心缓冲区，自动移除旧经验

    def push_all_batch(self, obs, action, logs, reward, next_obs, done, active=None):
        """将多个经验存储到缓冲区"""
        batch_size = obs.size(0)
        if active is None:
            active = torch.ones_like(done, dtype=torch.bool)
        for i in range(batch_size):
            # 打包所有数据为一个元组
            experience = (
                obs[i].clone().detach().to(DEVICE),
                action[i].clone().detach().to(DEVICE),
                logs[i].clone().detach().to(DEVICE),
                reward[i].clone().detach().to(DEVICE),
                next_obs[i].clone().detach().to(DEVICE),
                done[i].clone().detach().to(DEVICE),
                active[i].clone().detach().to(DEVICE),
            )
            self.buffer.append(experience)

    def push_batch(self, obs, action, logs, reward, next_obs, done, active=None):
        """将多个经验存储到缓冲区"""
        if active is None:
            active = torch.ones_like(done, dtype=torch.bool)
        experience = (
            obs.clone().detach().to(DEVICE),
            action.clone().detach().to(DEVICE),
            logs.clone().detach().to(DEVICE),
            reward.clone().detach().to(DEVICE),
            next_obs.clone().detach().to(DEVICE),
            done.clone().detach().to(DEVICE),
            active.clone().detach().to(DEVICE),
        )
        self.buffer.append(experience)
    
    def prepare_all_data(self):
        """返回整个缓冲区的数据"""
        return {
            'obs': torch.stack([exp[0] for exp in self.buffer]),
            'actions': torch.stack([exp[1] for exp in self.buffer]),
            'log_probs': torch.stack([exp[2] for exp in self.buffer]),
            'rewards': torch.stack([exp[3] for exp in self.buffer]),
            'next_obs': torch.stack([exp[4] for exp in self.buffer]),
            'dones': torch.stack([exp[5] for exp in self.buffer]),
            'active': torch.stack([exp[6] for exp in self.buffer]),
        }

    def clear(self):
        """清空所有存储的经验"""
        self.buffer.clear()

####################################    消息池    ####################################
class MessageBuffer:
    """消息池，用于存储和处理智能体间的消息"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 核心缓冲区，自动移除旧经验

    def push_batch(self, pre_message, next_message):
        """将多个经验存储到缓冲区"""
        batch_size = pre_message.size(0)
        for i in range(batch_size):
            # 打包所有数据为一个元组
            message = (
                pre_message[i].clone().detach().to(DEVICE),
                next_message[i].clone().detach().to(DEVICE)
            )
            self.buffer.append(message)
    
    def prepare_all_data(self):
        """返回整个缓冲区的数据"""
        return {
            'pre_message': torch.stack([exp[0] for exp in self.buffer]),
            'next_message': torch.stack([exp[1] for exp in self.buffer])
        }
    
    def clear(self):
        """清空所有存储的经验"""
        self.buffer.clear()
