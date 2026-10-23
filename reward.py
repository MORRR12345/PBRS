"""集中计算避障奖励，便于单独检查每一项。"""

import numpy as np

import config as cfg


def clearance(position):
    """智能体外缘到障碍物外缘的有符号净空距离。"""
    return float(np.linalg.norm(np.asarray(position) - cfg.OBSTACLE_CENTER)
                 - cfg.OBSTACLE_RADIUS - cfg.AGENT_RADIUS)

def potential(position):
    gap = max(0.0, cfg.SAFE_DISTANCE - clearance(position))
    return -cfg.SHAPING_WEIGHT * gap**2


def calculate_reward(reward_type, old_position, new_position, *, success=False,
                     collision=False, truncated=False):
    """返回总奖励及分项；只有到达目标才使用零终局势能。"""
    old_distance = np.linalg.norm(np.asarray(old_position) - cfg.GOAL)
    new_distance = np.linalg.norm(np.asarray(new_position) - cfg.GOAL)
    base = 10.0 * float(old_distance - new_distance) - 0.1
    goal_bonus = 20.0 if success else 0.0
    collision_cost = -10.0 if collision else 0.0
    shaping = 0.0
    if reward_type == "soft":
        shaping = potential(new_position)
    elif reward_type == "pbrs":
        next_potential = 0.0 if success else potential(new_position)
        shaping = cfg.GAMMA * next_potential - potential(old_position)
    elif reward_type != "hard":
        raise ValueError(f"未知奖励类型: {reward_type}")
    parts = {"base": base,
             "goal": goal_bonus,
             "collision": collision_cost,
             "shaping": shaping}
    return float(sum(parts.values())), parts
