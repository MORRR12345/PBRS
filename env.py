"""二维避障 Gymnasium 环境。"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as cfg
from reward import calculate_reward, clearance


class AvoidObstacleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_type="hard"):
        super().__init__()
        if reward_type not in {"hard", "soft", "pbrs"}:
            raise ValueError(f"未知奖励类型: {reward_type}")
        self.reward_type = reward_type
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # 位置、到目标向量、到障碍物向量、目标距离、障碍物净空距离
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.position = np.asarray(cfg.START, dtype=np.float32).copy()
        self.steps = 0

    def _clearance(self, position):
        return clearance(position)

    def _observe(self):
        goal_delta = np.asarray(cfg.GOAL, dtype=np.float32) - self.position
        obstacle_delta = np.asarray(cfg.OBSTACLE_CENTER, dtype=np.float32) - self.position
        return np.concatenate((self.position, goal_delta, obstacle_delta,
                               [np.linalg.norm(goal_delta), self._clearance(self.position)]))\
            .astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.asarray(cfg.START, dtype=np.float32).copy()
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        old_position = self.position.copy()
        self.position = self.position + cfg.MAX_SPEED * action
        self.steps += 1

        distance = float(np.linalg.norm(self.position - cfg.GOAL))
        collision = self._clearance(self.position) <= 0.0
        success = distance <= cfg.GOAL_THRESHOLD and not collision
        terminated = success
        truncated = self.steps >= cfg.MAX_STEPS and not terminated

        reward, reward_parts = calculate_reward(
            self.reward_type, old_position, self.position, success=success,
            collision=collision, truncated=truncated)

        info = {"success": bool(success), "collision": bool(collision),
                "reward_parts": reward_parts}
        return self._observe(), float(reward), bool(terminated), bool(truncated), info
