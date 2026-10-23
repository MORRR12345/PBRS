"""从训练日志生成三张对比图。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAMES = ("hard", "soft", "pbrs")


def smooth(values, window):
    """仅绘图时对每轮记录值做滑动平均。"""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        if np.isfinite(chunk).any():
            result[i] = np.nanmean(chunk)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30, help="return 滑动平均的采样轮数")
    parser.add_argument("--loss-window", type=int, default=5, help="loss 滑动平均的 PPO 更新次数")
    args = parser.parse_args()
    if args.window <= 0 or args.loss_window <= 0:
        parser.error("--window 和 --loss-window 必须大于零")
    Path("figures").mkdir(exist_ok=True)
    logs = {name: np.load(f"logs/{name}.npz") for name in NAMES}
    for key, title, filename in (("success", "Success rate", "success_rate.png"),
                                 ("returns", "Episode return", "episode_return.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, log in logs.items():
            rounds = np.arange(1, len(log[key]) + 1)
            values = np.cumsum(log[key]) / rounds if key == "success" else smooth(log[key], args.window)
            ax.plot(rounds, values, label=name)
        chart_title = "Cumulative success rate" if key == "success" else f"{title} ({args.window}-episode average)"
        ax.set(xlabel="Sampling round", ylabel=title, title=chart_title)
        if key == "success":
            ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figures/{filename}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, key, title in zip(axes, ("actor_loss", "critic_loss"),
                              ("Actor loss", "Critic loss")):
        for name, log in logs.items():
            values = log[key]
            updates = np.arange(1, len(values) + 1)
            ax.plot(updates, smooth(values, args.loss_window), label=name)
            ax.scatter(updates, values, s=8, alpha=0.15)
        ax.set(ylabel=title, title=f"{title} ({args.loss_window}-update average)")
        ax.legend()
    axes[-1].set_xlabel("PPO update / sampling round")
    fig.tight_layout()
    fig.savefig("figures/losses.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
