"""每轮让所有环境各完成一局，然后用合并的轨迹更新一次 PPO。"""

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm

import config as cfg
from env import AvoidObstacleEnv


def collect_round(model, envs):
    observations = [env.reset()[0] for env in envs]
    trajectories = [[] for _ in envs]
    returns = np.zeros(len(envs))
    collisions = np.zeros(len(envs), dtype=int)
    successes = np.zeros(len(envs), dtype=int)
    timeouts = np.zeros(len(envs), dtype=int)
    active = list(range(len(envs)))
    model.policy.set_training_mode(False)

    while active:
        with torch.no_grad():
            actions, values, log_probs = model.policy(torch.as_tensor(
                np.stack([observations[i] for i in active]), device=model.device))
        for j, i in enumerate(active[:]):
            action = actions[j].cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs[i].step(action)
            trajectories[i].append((observations[i], action, reward,
                                    values[j:j + 1], log_probs[j:j + 1]))
            returns[i] += reward
            collisions[i] += int(info["collision"])
            model.num_timesteps += 1
            observations[i] = next_obs
            if terminated or truncated:
                successes[i] = int(terminated)
                timeouts[i] = int(truncated)
                if truncated:
                    # 超时是采样边界，对最后一步加入末状态的 bootstrap 价值。
                    with torch.no_grad():
                        terminal_value = model.policy.predict_values(torch.as_tensor(
                            next_obs[None], device=model.device)).item()
                    state, act, raw_reward, value, log_prob = trajectories[i][-1]
                    trajectories[i][-1] = (state, act, raw_reward + cfg.GAMMA * terminal_value,
                                           value, log_prob)
                active.remove(i)

    size = sum(map(len, trajectories))
    buffer = RolloutBuffer(size, envs[0].observation_space, envs[0].action_space,
                           device=model.device, gamma=cfg.GAMMA, gae_lambda=model.gae_lambda)
    for trajectory in trajectories:
        for index, (state, action, reward, value, log_prob) in enumerate(trajectory):
            buffer.add(state[None], action[None], np.array([reward]),
                       np.array([index == 0]), value, log_prob)
    # 按环境拼接完整轨迹，episode_starts 在各轨迹之间切断 GAE。
    buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
    return buffer, returns, collisions, successes, timeouts, np.array(list(map(len, trajectories)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=cfg.TOTAL_EPISODES)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes 必须大于零")
    if cfg.NUM_ENVS <= 0:
        parser.error("NUM_ENVS 必须大于零")
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    for name in ("hard", "soft", "pbrs"):
        envs = [AvoidObstacleEnv(name) for _ in range(cfg.NUM_ENVS)]
        for i, env in enumerate(envs):
            env.reset(seed=cfg.SEED + i)
        model = PPO("MlpPolicy", envs[0], seed=cfg.SEED, gamma=cfg.GAMMA,
                    n_steps=cfg.MAX_STEPS, batch_size=cfg.MAX_STEPS, learning_rate=3e-4,
                    verbose=0, device="cpu")
        model.set_logger(configure(format_strings=[]))
        records = {key: [] for key in ("returns", "success", "collision", "timeout",
                                      "length", "actor_loss", "critic_loss")}
        for _ in tqdm(range(args.episodes), desc=name, unit="episode"):
            buffer, returns, collisions, successes, timeouts, lengths = collect_round(model, envs)
            model.rollout_buffer = buffer
            model._update_current_progress_remaining(model.num_timesteps,
                                                     args.episodes * cfg.MAX_STEPS * cfg.NUM_ENVS)
            model.train()
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
              f"avg_length={np.mean(records['length']):.1f}")
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
