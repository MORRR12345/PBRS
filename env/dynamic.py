"""二维质点动力学。"""

import torch


def update_pos(env, action, active=None):
    """按恒加速度更新二维位置和速度，已结束环境保持不动。"""
    with torch.no_grad():
        acc = env.max_a * clip_tensor(action, 0.0, 1.0)
        active_mask = None
        if active is not None:
            active_mask = active[:, None, None]
            acc = acc * active_mask

        dt = env.dec_step * env.sim_time
        env.vel += acc * dt
        displacement = env.vel * dt
        if active_mask is not None:
            displacement = displacement * active_mask
        env.pos += displacement

    return env.pos, env.vel


def clip_tensor(tensor, minimum, maximum):
    """按最后一维的向量范数进行裁剪。"""
    tensor_norm = torch.norm(tensor, dim=-1, keepdim=True)
    clipped_norm = torch.clamp(tensor_norm, minimum, maximum)
    return tensor * (clipped_norm / tensor_norm.clamp_min(1e-8))
