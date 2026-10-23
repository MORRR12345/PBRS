# python train_agent.py 训练agent强化学习部分

import time
import torch
import random

from env.environment import SpaceEnv
from agent.agent import Agent
from agent.buffer import ExperienceBuffer
from tool import get_savepath
from other.show import show_agent

#?#################################### 参数 ####################################?#
NUM_ENV = 64     # 环境数量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 100      # 最大训练epoch,正式训练改为1000或者3000、5000
SAVE_EPOCHS = 50      # 保存策略间隔epoch,正式训练改为100或者500
SHOW_EPOCHS = 10       # 打印奖励间隔epoch

# 设置随机种子
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    random.seed(seed)

#?#################################### 初始化 ####################################?#

def init():
    """完成所有初始化"""
    print("👊 Start Init......")
    env = SpaceEnv(NUM_ENV) # 环境
    agent = Agent(env.state_dim, env.act_dim, DEVICE) # 智能体

    expe_buffer = ExperienceBuffer(NUM_ENV*env.max_step) # 经验回放池

    print("👌 Init done")

    return env, agent, expe_buffer

#?#################################### 训练agent ####################################?#

def train_agent(env, agent, expe_buffer):
    """训练agent"""
    print("⛳ Start Train Actor......")
    set_seed(1)

    # 记录用于监控训练
    actor_losses = []
    critic_losses = []
    reward_history = []
    best_reward = -1e9

    # 准备工作
    save_path = get_savepath()
    
    init_time = time.time()

    #*#################################### train开始 ####################################*#
    
    for epoch in range(MAX_EPOCHS+1):
        obs, done = env.reset()
        expe_buffer.clear()
        episode_reward = torch.zeros(env.num_env, 2, device=DEVICE)

        #*#################################### 收集数据 ####################################*#

        #* 1、收集一批数据
        while not torch.all(done):
            active = ~done
            # 1、生成动作
            actions, _, log_probs = agent.get_action(obs) # [N,n,obs] -> [N,n,action]

            # 2、执行动作
            next_obs, reward, done = env.step(actions) # [N,n,action] -> [N,reward]

            # 3、存储经验
            expe_buffer.push_batch(
                obs,                             # obs:       [N,n,obs]
                actions,                         # act:       [N,n,action]
                log_probs,                       # log_probs: [N]
                torch.sum(reward, dim=-1),       # reward:    [N]
                next_obs,                        # next_obs:  [N,n,next_obs]
                done,                            # done:      [N]
                active,                          # active:    [N]
            )

            # 4、处理
            obs = next_obs.detach().clone()
            episode_reward += reward

        #*#################################### 收集数据结束 ####################################*#
        mean_reward = torch.mean(episode_reward, dim=0)
        reward_history.append(mean_reward) # 记录所有环境的平均奖励

        #* 2、保存目前最好的模型
        score = torch.sum(mean_reward).item()
        if score > best_reward and epoch > int(MAX_EPOCHS/2):
            best_reward = score
            torch.save(agent.actor.state_dict(), f"{save_path}/actor_pth/best_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_path}/critic_pth/best_critic.pth")

            torch.save(torch.stack(reward_history), f"{save_path}/data/best_reward_history.pt")
            torch.save(actor_losses, f"{save_path}/data/best_actor_losses.pt")
            torch.save(critic_losses, f"{save_path}/data/best_critic_losses.pt")

        #* 3、更新actor和critic
        expe_data = expe_buffer.prepare_all_data()
        actor_loss, critic_loss = agent.update_actor_critic(expe_data)
        actor_losses.append(actor_loss) # 记录
        critic_losses.append(critic_loss) # 记录

        #* 4、定期显示训练结果
        if epoch % SHOW_EPOCHS == 0:
            use_time_epoch = (time.time()-init_time)/SHOW_EPOCHS
            init_time = time.time()
            print(f"🕒 Epoch {epoch}|{MAX_EPOCHS}: "
                  f"R:{torch.sum(mean_reward).item():.2f}, A:{actor_loss:.4f}, C:{critic_loss:.2f}, "
                  f"{use_time_epoch:.2f} S/epoch, {use_time_epoch*epoch/60:.2f}|{use_time_epoch*MAX_EPOCHS/60:.2f}min")

        #* 5、定期保存模型
        if epoch % SAVE_EPOCHS == 0:
            torch.save(agent.actor.state_dict(), f"{save_path}/actor_pth/actor_epoch_{epoch}.pth")
            torch.save(agent.critic.state_dict(), f"{save_path}/critic_pth/critic_epoch_{epoch}.pth")

            torch.save(torch.stack(reward_history), f"{save_path}/data/reward_history.pt")
            torch.save(actor_losses, f"{save_path}/data/actor_losses.pt")
            torch.save(critic_losses, f"{save_path}/data/critic_losses.pt")
            print("Save Done")
            torch.cuda.empty_cache()#清理缓存
    
    #*#################################### train结束 ####################################*#
    show_agent(save_path)

    return best_reward

if __name__ == "__main__":
    env, agent, expe_buffer = init()
    #* 训练agent
    best_reward = train_agent(env, agent, expe_buffer)
    #* 结束
    print("👍 Training completed!, 🥇 Best Reward:", best_reward)
