"""二维导航任务的奖励函数。"""

import torch


def compute_reward(env, active, reached, collision):
    """返回 [导航奖励, 控制奖励]，形状为 [N, 2]。"""
    with torch.no_grad():
        error = torch.norm(env.pos - env.target_pos, dim=-1).mean(dim=-1)
        navigation_reward = -error / env.map_size

        # 越靠近障碍物惩罚越强；碰撞和到达目标使用明确的终止奖励。
        obstacle_distance = torch.norm(
            env.pos - env.obstacle_center.view(1, 1, 2), dim=-1
        ).amin(dim=-1)
        clearance = (obstacle_distance - env.obstacle_radius) / env.map_size
        navigation_reward -= 0.5 * torch.exp(-5.0 * clearance.clamp_min(0.0))
        navigation_reward += reached.float() * 20.0
        navigation_reward -= collision.float() * 20.0

        control_reward = env.fuel[:, :, 0].mean(dim=-1) - 1.0

        reward = torch.stack((navigation_reward, control_reward), dim=-1)
        
        return reward * active[:, None]
