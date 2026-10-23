"""实验参数；修改后需重新训练模型。"""

WORLD_SIZE = 10.0  # 绘图参考范围；智能体位置不受该范围限制
START = (1.0, 5.0)  # 智能体固定起点 (x, y)
GOAL = (9.0, 5.0)  # 固定目标点 (x, y)
OBSTACLE_CENTER = (5.0, 5.0)  # 圆形障碍物中心 (x, y)
OBSTACLE_RADIUS = 1.1  # 障碍物半径
AGENT_RADIUS = 0.15  # 智能体半径；障碍物碰撞时计入
MAX_STEPS = 100  # 每局最大步数，达到后截断
GOAL_THRESHOLD = 0.2  # 智能体中心到目标不超过该距离即成功
MAX_SPEED = 0.2  # 动作每个坐标分量对应的最大单步位移
SEED = 42  # 环境与 PPO 共用的随机种子

GAMMA = 0.99  # PPO 折扣因子，同时用于 PBRS 势能差
SAFE_DISTANCE = 1.5  # 智能体外缘到障碍物外缘的安全距离
SHAPING_WEIGHT = 1.0  # soft 惩罚和 PBRS 势能的权重
TOTAL_EPISODES = 100  # 每种奖励方案的采样轮数；每轮每个环境完成一局并更新一次
NUM_ENVS = 32  # 每轮同时采样的环境数
