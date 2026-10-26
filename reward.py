"""批量计算避障奖励及 PBRS 势能。"""

import torch

import config as cfg


def torch_potential_from_clearance(clearance_values):
    """由已计算的净空距离得到势能，避免重复计算位置距离。"""
    gap = torch.clamp(cfg.SAFE_DISTANCE - clearance_values, min=0.0)
    return -cfg.SHAPING_WEIGHT * gap.square()


def torch_reward(reward_type, old_distance, new_distance, old_clearance,
                 new_clearance, success, collision, return_parts=True):
    """复用观测中的距离，批量计算各环境的奖励分项。"""
    base = 10.0 * (old_distance - new_distance) - 0.1
    goal_bonus = success.to(base.dtype) * 20.0
    collision_cost = collision.to(base.dtype) * -10.0
    if reward_type == "soft":
        shaping = torch_potential_from_clearance(new_clearance)
    elif reward_type == "pbrs":
        next_potential = torch.where(success, 0.0, torch_potential_from_clearance(new_clearance))
        shaping = cfg.GAMMA * next_potential - torch_potential_from_clearance(old_clearance)
    elif reward_type == "hard":
        shaping = torch.zeros_like(base)
    else:
        raise ValueError(f"未知奖励类型: {reward_type}")
    total = base + goal_bonus + collision_cost + shaping
    if not return_parts:
        return total, None
    parts = {"base": base, "goal": goal_bonus, "collision": collision_cost,
             "shaping": shaping}
    return total, parts
