"""基于 Torch 的二维避障批量环境，以及供 Gymnasium 使用的单环境接口。"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import config as cfg
from reward import torch_reward


class TorchAvoidObstacleEnv:
    """在一个张量中同时维护多个独立的避障环境。"""

    def __init__(self, num_envs, reward_type="hard"):
        """在 CUDA 上初始化指定数量的环境及奖励方案。"""
        if num_envs <= 0:
            raise ValueError("num_envs 必须大于零")
        if reward_type not in {"hard", "soft", "pbrs"}:
            raise ValueError(f"未知奖励类型: {reward_type}")
        self.num_envs = num_envs
        self.reward_type = reward_type
        self.device = torch.device("cuda")
        self.start = torch.tensor(cfg.START, dtype=torch.float32, device=self.device)
        self.goal = torch.tensor(cfg.GOAL, dtype=torch.float32, device=self.device)
        self.obstacle = torch.tensor(cfg.OBSTACLE_CENTER, dtype=torch.float32, device=self.device)
        self.reset()

    def reset(self):
        """同时重置全部环境并返回形状为 (环境数, 8) 的观测。"""
        self.positions = self.start.expand(self.num_envs, -1).clone()
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._observation = self._make_observation()
        return self._observation

    def _make_observation(self):
        """批量拼接位置、相对向量、目标距离和障碍物净空距离。"""
        goal_delta = self.goal - self.positions
        obstacle_delta = self.obstacle - self.positions
        clearance = (torch.linalg.vector_norm(obstacle_delta, dim=-1, keepdim=True)
                     - cfg.OBSTACLE_RADIUS - cfg.AGENT_RADIUS)
        return torch.cat((self.positions, goal_delta, obstacle_delta,
                          torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True),
                          clearance), dim=-1)

    def step(self, actions, *, reward_parts=True):
        """同时推进活动环境；训练时可省略奖励分项以减少张量操作。"""
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"动作形状应为 ({self.num_envs}, 2)，实际为 {tuple(actions.shape)}")
        active = ~self.done
        old_observation = self._observation
        moved = self.positions + cfg.MAX_SPEED * actions.clamp(-1.0, 1.0)
        self.positions = torch.where(active.unsqueeze(-1), moved, self.positions)
        self.steps += active.long()

        next_observation = self._make_observation()
        collision = active & (next_observation[:, 7] <= 0.0)
        success = active & (next_observation[:, 6] <= cfg.GOAL_THRESHOLD) & ~collision
        truncated = active & (self.steps >= cfg.MAX_STEPS) & ~success
        reward, parts = torch_reward(self.reward_type, old_observation[:, 6],
                                     next_observation[:, 6], old_observation[:, 7],
                                     next_observation[:, 7], success, collision,
                                     return_parts=reward_parts)
        reward = torch.where(active, reward, 0.0)
        if parts is not None:
            parts = {key: torch.where(active, value, 0.0) for key, value in parts.items()}
        self.done |= success | truncated
        self._observation = next_observation
        info = {"success": success, "collision": collision}
        if parts is not None:
            info["reward_parts"] = parts
        return next_observation, reward, success, truncated, info


class AvoidObstacleEnv(gym.Env):
    """将单个 Torch 环境包装为 Gymnasium 接口，供 PPO 建模和回放。"""

    metadata = {"render_modes": []}

    def __init__(self, reward_type="hard"):
        """初始化单环境张量状态及 Gymnasium 动作、观测空间。"""
        super().__init__()
        self.batch = TorchAvoidObstacleEnv(1, reward_type)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # 位置、到目标向量、到障碍物向量、目标距离、障碍物净空距离。
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

    @property
    def position(self):
        """返回供路径绘图使用的单环境位置。"""
        return self.batch.positions[0].cpu().numpy().copy()

    def reset(self, *, seed=None, options=None):
        """重置环境并返回 Gymnasium 格式的观测与信息。"""
        super().reset(seed=seed)
        return self.batch.reset()[0].cpu().numpy().copy(), {}

    def step(self, action):
        """推进一个环境并将批量张量结果转换为 Gymnasium 返回值。"""
        observation, reward, terminated, truncated, info = self.batch.step(
            torch.as_tensor(action, dtype=torch.float32, device=self.batch.device).reshape(1, 2))
        reward_parts = {key: float(value[0]) for key, value in info["reward_parts"].items()}
        return (observation[0].cpu().numpy().copy(), float(reward[0]),
                bool(terminated[0]), bool(truncated[0]),
                {"success": bool(info["success"][0]),
                 "collision": bool(info["collision"][0]), "reward_parts": reward_parts})
