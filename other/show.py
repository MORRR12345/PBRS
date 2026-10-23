# 展示训练曲线

import os
import re
import torch
import matplotlib.pyplot as plt # type: ignore

def _get_time_path(base_path, time):
    """获取时间目录路径"""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"基础路径不存在: {base_path}")
    
    if time == "latest":
        time_dirs = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d)) and 
                    re.match(r"Train_data_\d{2}-\d{2}-\d{2}-\d{2}", d)]
        
        if not time_dirs:
            raise FileNotFoundError(f"在 {base_path} 中未找到时间目录")
            
        # 按创建时间排序获取最新的
        time_dirs.sort(key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
        return os.path.join(base_path, time_dirs[0])
    
    else:
        time_dir = f"Train_data_{time}"
        path = os.path.join(base_path, time_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"时间目录不存在: {path}")
        return path

def show_agent(save_path):
    """显示最新的训练曲线"""
    reward_history = torch.load(f"{save_path}/data/reward_history.pt")
    actor_losses = torch.load(f"{save_path}/data/actor_losses.pt")
    critic_losses = torch.load(f"{save_path}/data/critic_losses.pt")

    plt.figure(figsize=(12, 8))

    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(torch.sum(reward_history,dim=-1).cpu().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Total Reward')

    # 奖励曲线
    plt.subplot(2, 2, 2)
    plt.plot(reward_history[:,0].cpu().numpy(), label='Navigation Reward')
    plt.plot(reward_history[:,1].cpu().numpy(), label='Control Reward')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('End Reward')
    plt.title('Detal Reward')

    # 损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(actor_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Actor Loss')

    plt.subplot(2, 2, 4)
    plt.plot(critic_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Critic Loss')

    # 保存图表
    plt.tight_layout()
    plt.savefig(f"{save_path}/picture/training_AGENT.pdf", format='pdf') 
    plt.show()
    plt.close()


if __name__ == "__main__":
    save_path = _get_time_path("model", "latest")
    show_agent(save_path)

