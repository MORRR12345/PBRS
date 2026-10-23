# python play.py 运行训练好的模型进行仿真演示

import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt # type: ignore

from agent.agent import Agent
from env.visual import Visualizer
from env.environment import SpaceEnv
from tool import _get_time_path

# 需要准备的参数
NUM_AGENTS = 1  # 单个二维导航智能体

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 完成所有初始化
def init():
    """完成所有初始化"""
    print("👊 Start Init......")

    env = SpaceEnv(1, NUM_AGENTS)
    vis = Visualizer(
        env.map_size,
        NUM_AGENTS,
        env.total_time,
        env.obstacle_center.cpu().numpy(),
        env.obstacle_radius,
    )

    obs_dim, act_dim, state_dim, num_dim = env.get_dim()
    agent = Agent(obs_dim, act_dim, state_dim, num_dim, DEVICE) # 智能体
    agent.load("actor", time="latest", epoch="best") # best

    path = _get_time_path("model", "latest")

    print("👌 Init done")

    return env, vis, agent, path

# 主训练循环
def play(env, agent):
    """主训练循环"""
    set_seed(1)

    # 记录变量
    h_pos, h_action, h_reward, h_fuel = [], [], [], []
    h_time, h_error = [], []
    # 重置环境
    obs, done = env.reset()
    
    ####################################################################################

    # 收集一个回合的数据
    while not torch.all(done):
        #* 1. 生成动作
        _, actions, _ = agent.get_action(obs) # [1,n,obs] -> [1,n,action]

        #* 2. 执行动作
        next_obs, reward, done = env.step(actions) # [N,action] -> [N,obs]

        #* 3. 保存信息
        h_pos.append(env.pos[0].clone()) # 记录第一个环境中的二维位置
        h_action.append(actions[0]) # 记录第0个智能体的动作
        for _ in range(env.dec_step-1):
            h_action.append(torch.zeros_like(actions[0]))
        h_reward.append(reward[0].clone()) # 记录第一个环境的奖励分量
        h_fuel.append(env.fuel[0].clone()) # 记录燃料
        h_time.append(env.time)
        
        #* 5. 更新上一时刻的消息和动作
        obs = next_obs

    record = {
        "h_pos": torch.stack(h_pos),
        "h_action": torch.stack(h_action),
        "h_reward": torch.stack(h_reward),
        "h_fuel": torch.stack(h_fuel),
        "target_pos": env.target_pos[0],
        "obstacle_center": env.obstacle_center,
        "obstacle_radius": env.obstacle_radius,
        "h_time": torch.tensor(h_time)
    }

    torch.save(record, "record.pt")
    return record

# 主训练循环
def show(vis, path):
    """主训练循环"""
    record = torch.load("record.pt")

    h_pos = record["h_pos"].detach().cpu().numpy()
    h_action = record["h_action"].detach().cpu().numpy()
    h_reward = record["h_reward"].detach().cpu().numpy()
    h_fuel = record["h_fuel"].detach().cpu().numpy()
    target_pos = record["target_pos"].detach().cpu().numpy()
    h_time = record["h_time"].detach().cpu().numpy()

    # 初始化
    agent_scatter = vis.init_scatter(h_pos[0], target_pos)

    # 演示函数
    for step in tqdm(range(1, h_pos.shape[0])):
        #* 1. 检查窗口是否关闭
        if not plt.fignum_exists(vis.fig_2d.number):
            print("窗口已关闭，程序自动结束。")
            break
        #* 2. 更新位置、动作和奖励
        vis.move_scatter(agent_scatter, h_pos[step])
        vis.show_history(h_pos[step], h_pos[step-1])

        plt.draw()
        plt.pause(0.01)  # 更流畅的动画

    vis.show_reward(h_reward, h_time)
    vis.show_action(h_action)
    vis.save(path)
    # 信息显示
    error = np.linalg.norm(h_pos[-1] - target_pos, axis=-1)
    print(f'🕒 Time:{h_time[-1]:.0f}   🚀 Agent:{NUM_AGENTS} \n'
          f'🏆 Reward:{np.sum(h_reward[-1]):.2f}   🎯 Error:{np.mean(error):.2f}±{np.std(error):.2f} \n')

if __name__ == "__main__":
    env, vis, agent, path = init()

    record = play(env, agent)
    
    show(vis, path)
