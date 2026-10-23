"""二维环境可视化。"""

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Circle


class Visualizer:
    """二维导航轨迹渲染器。"""

    def __init__(
        self,
        map_size,
        num_agents,
        max_time,
        obstacle_center=(0.0, 0.0),
        obstacle_radius=200.0,
    ):
        self.map_size = map_size
        self.num_agents = num_agents
        self.max_time = max_time

        self.fig_curves = plt.figure(figsize=(6, 6))
        self.fig_2d = plt.figure(figsize=(6, 6))
        self.ax_reward = self.fig_curves.add_subplot(211)
        self.ax_action = self.fig_curves.add_subplot(212)
        self.fig_curves.subplots_adjust(hspace=0.4)
        self.ax_2d = self.fig_2d.add_subplot(111)

        self.ax_2d.set_xlim(-map_size, map_size)
        self.ax_2d.set_ylim(-map_size, map_size)
        self.ax_2d.set_aspect("equal", adjustable="box")
        self.ax_2d.set_xlabel("X (m)", fontsize=10)
        self.ax_2d.set_ylabel("Y (m)", fontsize=10)
        self.ax_2d.set_title("2D navigation trajectory")
        self.ax_2d.grid(True)
        self.ax_2d.add_patch(
            Circle(
                obstacle_center,
                obstacle_radius,
                facecolor="#808080",
                edgecolor="#303030",
                alpha=0.45,
                label="Obstacle",
            )
        )

        rgb_array = np.random.default_rng(0).integers(
            80, 230, size=(num_agents, 3)
        )
        self.history_colors = [
            f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_array
        ]

        plt.ion()
        plt.draw()
        plt.show()

    def init_scatter(self, agent_pos, target_pos):
        self.ax_2d.scatter(
            agent_pos[:, 0], agent_pos[:, 1],
            c="#000000", s=20, marker="o", label="Start",
        )
        self.ax_2d.scatter(
            target_pos[:, 0], target_pos[:, 1],
            s=35, marker="o", edgecolors="#F96464", facecolors="none",
            linewidth=1.2, label="Goal",
        )
        agent_scatter = self.ax_2d.scatter(
            agent_pos[:, 0], agent_pos[:, 1],
            c="#004CFF", s=18, marker="D", label="Agent",
        )
        self.ax_2d.legend(loc="upper right", fontsize=8)
        return agent_scatter

    def move_scatter(self, scatter, pos):
        scatter.set_offsets(pos[:, :2])

    def show_history(self, pos1, pos2):
        for i in range(pos1.shape[0]):
            self.ax_2d.plot(
                [pos1[i, 0], pos2[i, 0]],
                [pos1[i, 1], pos2[i, 1]],
                color=self.history_colors[i],
                linewidth=1,
            )

    def show_reward(self, reward, time):
        self.ax_reward.set_xlabel("Time (s)", fontsize=10)
        self.ax_reward.set_ylabel("Reward", fontsize=10)
        self.ax_reward.grid(True)
        reward = np.asarray(reward)
        labels = ("Navigation", "Control")
        if reward.ndim == 1:
            self.ax_reward.plot(time, reward, linewidth=1, label="Reward")
        else:
            for i in range(reward.shape[-1]):
                label = labels[i] if i < len(labels) else f"Reward {i}"
                self.ax_reward.plot(time, reward[:, i], linewidth=1, label=label)
        self.ax_reward.set_xlim(0, time[-1])
        self.ax_reward.legend(loc="lower right", fontsize=8)

    def show_action(self, action):
        self.ax_action.set_xlabel("Step", fontsize=10)
        self.ax_action.set_ylabel("Action", fontsize=10)
        self.ax_action.grid(True)
        action = np.asarray(action)
        if action.ndim == 3:
            action = action[:, 0, :]
        for i, label in enumerate(("ax", "ay")):
            self.ax_action.plot(action[:, i], linewidth=1, label=label)
        self.ax_action.set_xlim(0, len(action) - 1)
        self.ax_action.legend(loc="upper right", fontsize=8)

    def show_collision(self, agent_scatter, warn_mask, collision_mask):
        colors = np.array(["#004CFF"] * len(warn_mask))
        colors[warn_mask] = "#FDDA58"
        colors[collision_mask] = "#FF5252"
        agent_scatter.set_color(colors)

    def clear_all(self):
        for line in list(self.ax_2d.lines):
            line.remove()
        for collection in list(self.ax_2d.collections):
            collection.remove()

    def save(self, path):
        self.fig_2d.savefig(
            f"{path}/picture/play_2d.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.3,
        )
        self.fig_curves.savefig(
            f"{path}/picture/play_curves.pdf", format="pdf", bbox_inches="tight"
        )
