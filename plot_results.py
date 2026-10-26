"""从训练日志生成三张对比图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as cfg

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
    """读取三种奖励的训练日志，绘制成功率、回报和损失对比图。"""
    if cfg.RETURN_SMOOTH_WINDOW <= 0 or cfg.LOSS_SMOOTH_WINDOW <= 0:
        raise ValueError("RETURN_SMOOTH_WINDOW 和 LOSS_SMOOTH_WINDOW 必须大于零")
    Path("figures").mkdir(exist_ok=True)
    logs = {name: np.load(f"logs/{name}.npz") for name in NAMES}
    for key, title, filename in (("success", "Success rate", "success_rate.png"),
                                 ("returns", "Episode return", "episode_return.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, log in logs.items():
            rounds = np.arange(1, len(log[key]) + 1)
            # 成功率显示从首轮起的累计平均值；回报使用指定窗口平滑。
            values = (np.cumsum(log[key]) / rounds if key == "success"
                      else smooth(log[key], cfg.RETURN_SMOOTH_WINDOW))
            ax.plot(rounds, values, label=name)
        chart_title = ("Cumulative success rate" if key == "success"
                       else f"{title} ({cfg.RETURN_SMOOTH_WINDOW}-episode average)")
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
            line, = ax.plot(updates, smooth(values, cfg.LOSS_SMOOTH_WINDOW), label=name)
            ax.plot(updates, values, color=line.get_color(), alpha=0.2)
        ax.set(ylabel=title, title=f"{title} ({cfg.LOSS_SMOOTH_WINDOW}-update average)")
        ax.legend()
    axes[-1].set_xlabel("PPO update / sampling round")
    fig.tight_layout()
    fig.savefig("figures/losses.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
