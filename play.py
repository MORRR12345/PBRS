"""确定性回放并绘制路径。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from stable_baselines3 import PPO

import config as cfg
from env import AvoidObstacleEnv


def rollout(name):
    env = AvoidObstacleEnv(name)
    model = PPO.load(f"models/ppo_{name}.zip", device="cpu")
    obs, _ = env.reset(seed=cfg.SEED)
    path = [env.position.copy()]
    info = {}
    collision_steps = 0
    for _ in range(cfg.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        path.append(env.position.copy())
        collision_steps += int(info["collision"])
        if terminated or truncated:
            break
    env.close()
    status = "success" if info.get("success") else "timeout"
    if collision_steps:
        status += f", collision steps={collision_steps}"
    return np.asarray(path), status


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reward", choices=("hard", "soft", "pbrs"))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = ("hard", "soft", "pbrs") if args.all else (args.reward,)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Circle(cfg.OBSTACLE_CENTER, cfg.OBSTACLE_RADIUS,
                        color="tab:red", alpha=0.35, label="obstacle"))
    ax.add_patch(Circle(cfg.OBSTACLE_CENTER, cfg.OBSTACLE_RADIUS + cfg.AGENT_RADIUS,
                        fill=False, edgecolor="tab:red", linestyle="--", alpha=0.6))
    ax.scatter(*cfg.START, marker="o", color="green", s=80, label="start", zorder=5)
    ax.scatter(*cfg.GOAL, marker="*", color="blue", s=140, label="goal", zorder=5)
    paths = []
    for name in names:
        path, status = rollout(name)
        paths.append(path)
        line, = ax.plot(path[:, 0], path[:, 1], label=f"{name}: {status}")
        ax.scatter(*path[-1], color=line.get_color(), marker="x", s=60)
        print(f"{name}: {status}, steps={len(path) - 1}")
    points = np.concatenate(paths)
    ax.set(xlim=(min(0, points[:, 0].min() - 0.5), max(cfg.WORLD_SIZE, points[:, 0].max() + 0.5)),
           ylim=(min(0, points[:, 1].min() - 0.5), max(cfg.WORLD_SIZE, points[:, 1].max() + 0.5)),
           xlabel="x", ylabel="y", title="Deterministic PPO paths")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/play_paths.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
