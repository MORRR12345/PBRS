"""每轮让所有环境各完成一局，然后用合并的轨迹更新一次 PPO。"""

import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm

import config as cfg
from env import AvoidObstacleEnv, TorchAvoidObstacleEnv


def collect_round(model, env, observation_space, action_space):
    """并行采集每个环境的一局，批量写入 PPO 缓冲区并返回统计。"""
    observations = env.reset()
    count = env.num_envs
    device = env.device
    shape = (cfg.MAX_STEPS, count)
    states = torch.empty((*shape, *observation_space.shape), dtype=torch.float32, device=device)
    actions_taken = torch.empty((*shape, *action_space.shape), dtype=torch.float32, device=device)
    rewards = torch.empty(shape, dtype=torch.float32, device=device)
    values = torch.empty(shape, dtype=torch.float32, device=device)
    log_probs = torch.empty(shape, dtype=torch.float32, device=device)
    returns = torch.zeros(count, dtype=torch.float32, device=device)
    collisions = torch.zeros(count, dtype=torch.long, device=device)
    successes = torch.zeros(count, dtype=torch.bool, device=device)
    timeouts = torch.zeros(count, dtype=torch.bool, device=device)
    model.policy.set_training_mode(False)

    with torch.inference_mode():
        for step in range(cfg.MAX_STEPS):
            policy_actions, policy_values, policy_log_probs = model.policy(observations)
            next_observations, reward, terminated, truncated, info = env.step(
                policy_actions, reward_parts=False)
            returns += reward  # 日志保留环境原始回报，不包含训练用的超时 bootstrap。
            states[step] = observations
            actions_taken[step] = policy_actions
            rewards[step] = reward
            values[step] = policy_values.flatten()
            log_probs[step] = policy_log_probs.flatten()
            collisions += info["collision"].long()
            successes |= terminated
            timeouts |= truncated
            observations = next_observations
            # 每 10 步检查一次是否全部结束，减少 GPU 上的逐步同步。
            if (step + 1) % 10 == 0 and bool(env.done.all()):
                break

        if step == cfg.MAX_STEPS - 1 and bool(truncated.any()):
            # 超时只会发生在最后一步，训练奖励加入末状态价值。
            terminal_values = model.policy.predict_values(observations[truncated]).flatten()
            rewards[step, truncated] += cfg.GAMMA * terminal_values

    lengths = env.steps
    size = int(lengths.sum().item())
    model.num_timesteps += size
    buffer = RolloutBuffer(size, observation_space, action_space, device=model.device,
                           gamma=cfg.GAMMA, gae_lambda=model.gae_lambda)
    # 按环境拼接完整轨迹；布尔掩码剔除已结束环境的填充步。
    valid = torch.arange(step + 1, device=device)[:, None] < lengths[None, :]

    def flatten(tensor):
        """将时间优先的张量按环境顺序展平，只保留有效步。"""
        return tensor[:step + 1].transpose(0, 1)[valid.T].cpu().numpy()

    buffer.observations[:, 0] = flatten(states)
    buffer.actions[:, 0] = flatten(actions_taken)
    buffer.rewards[:, 0] = flatten(rewards)
    buffer.values[:, 0] = flatten(values)
    buffer.log_probs[:, 0] = flatten(log_probs)
    starts = torch.zeros_like(rewards[:step + 1])
    starts[0] = 1.0
    buffer.episode_starts[:, 0] = flatten(starts)
    buffer.pos = size
    buffer.full = True
    # 各轨迹首步的 episode_starts 会在 GAE 中切断前一个环境的轨迹。
    buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
    return (buffer, returns.cpu().numpy(), collisions.cpu().numpy(),
            successes.cpu().numpy(), timeouts.cpu().numpy(), lengths.cpu().numpy())


def main():
    """按三种奖励分别训练 PPO，并保存模型与每轮统计日志。"""
    if cfg.TOTAL_EPISODES <= 0:
        raise ValueError("TOTAL_EPISODES 必须大于零")
    if cfg.NUM_ENVS <= 0:
        raise ValueError("NUM_ENVS 必须大于零")
    if cfg.MAX_STEPS <= 1 or cfg.PPO_EPOCHS <= 0 or cfg.PPO_BATCH_SIZE <= 1:
        raise ValueError("MAX_STEPS、PPO_EPOCHS 和 PPO_BATCH_SIZE 的配置无效")
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    for name in ("hard", "soft", "pbrs"):
        reference_env = AvoidObstacleEnv(name)
        # 已明确使用 CUDA；仅屏蔽 SB3 对小型 MLP 的通用性能提醒。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"You are trying to run PPO on the GPU",
                category=UserWarning, module=r"stable_baselines3\.common\.on_policy_algorithm")
            model = PPO("MlpPolicy", reference_env, seed=cfg.SEED, gamma=cfg.GAMMA,
                        n_steps=cfg.MAX_STEPS, batch_size=cfg.MAX_STEPS,
                        n_epochs=cfg.PPO_EPOCHS, learning_rate=3e-4, verbose=0,
                        device="cuda")
        # 构造时仅有一个参考环境；实际训练缓冲区来自 NUM_ENVS 个环境。
        model.batch_size = cfg.PPO_BATCH_SIZE
        model.set_logger(configure(format_strings=[]))
        env = TorchAvoidObstacleEnv(cfg.NUM_ENVS, name)
        # 每条记录对应一轮；成功率等指标先在本轮环境之间取平均。
        records = {key: [] for key in ("returns", "success", "collision", "timeout",
                                      "length", "actor_loss", "critic_loss")}
        sampling_seconds = 0.0
        optimization_seconds = 0.0
        for _ in tqdm(range(cfg.TOTAL_EPISODES), desc=name, unit="episode"):
            started = perf_counter()
            buffer, returns, collisions, successes, timeouts, lengths = collect_round(
                model, env, reference_env.observation_space, reference_env.action_space)
            sampling_seconds += perf_counter() - started
            model.rollout_buffer = buffer
            # 按已采集步数更新学习率进度，再进行一次 PPO 优化。
            model._update_current_progress_remaining(model.num_timesteps,
                                                     cfg.TOTAL_EPISODES * cfg.MAX_STEPS * cfg.NUM_ENVS)
            started = perf_counter()
            model.train()
            optimization_seconds += perf_counter() - started
            losses = model.logger.name_to_value
            records["returns"].append(float(np.mean(returns)))
            records["success"].append(float(np.mean(successes)))
            records["collision"].append(float(np.mean(collisions)))
            records["timeout"].append(float(np.mean(timeouts)))
            records["length"].append(float(np.mean(lengths)))
            records["actor_loss"].append(losses["train/policy_gradient_loss"])
            records["critic_loss"].append(losses["train/value_loss"])

        model.save(f"models/ppo_{name}")
        np.savez(f"logs/{name}.npz", num_envs=cfg.NUM_ENVS,
                 **{key: np.asarray(value) for key, value in records.items()})
        print(f"{name}: success={np.mean(records['success']):.1%}, "
              f"collision steps={np.mean(records['collision']):.1f}, "
              f"avg_length={np.mean(records['length']):.1f}, "
              f"sampling={sampling_seconds:.1f}s, ppo={optimization_seconds:.1f}s")
        reference_env.close()


if __name__ == "__main__":
    main()
