"""二维并行导航环境。"""

import torch

from env.dynamic import update_pos
from env.reward import compute_reward

MAP_SIZE = 1000.0
POS_SCALE = 1.0 / MAP_SIZE
VEL_SCALE = 1.0

TOTAL_TIME = 1000.0
SIM_TIME = 1.0
DEC_STEP = 10
MAX_A = 0.1

START_POS = [-800.0, 0.0] # 起始位置
END_POS = [800.0, 0.0] # 目标位置
OBSTACLE_POS = [0.0, 0.0] # 障碍物位置
END_RADIUS = 30.0 # 目标半径
OBSTACLE_RADIUS = 200.0 # 障碍物半径

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpaceEnv:
    """N 个并行、具有相同场景参数的二维环境。"""

    def __init__(self, num_env):
        self.num_env = int(num_env)

        # 位置2 + 速度2 + 障碍物中心2 + 障碍物半径1 + 目标位置2
        self.state_dim = 9
        self.action_dim = 2

        self.total_time = TOTAL_TIME
        self.sim_time = SIM_TIME
        self.dec_step = DEC_STEP
        self.max_step = int(self.total_time / (self.dec_step * self.sim_time))
        self.map_size = MAP_SIZE
        self.max_a = MAX_A
        self.end_radius = END_RADIUS

        #* 起点、终点和障碍物
        self.end = torch.zeros(self.num_env, 2, device=DEVICE, dtype=torch.float32)
        self.obstacle = torch.zeros(self.num_env, 3, device=DEVICE, dtype=torch.float32)

        #* 位置、速度、done标志和时间
        self.pos = torch.zeros(self.num_env, 2, device=DEVICE, dtype=torch.float32)
        self.vel = torch.zeros(self.num_env, 2, device=DEVICE, dtype=torch.float32)
        self.done = torch.zeros(self.num_env, device=DEVICE, dtype=torch.bool)
        self.time = 0.0

    def get_state(self):
        return torch.cat(
            (
                self.pos, # 位置
                self.vel, # 速度
                self.obstacle, # 障碍物中心和半径
                self.end # 目标位置
            ),
            dim=-1,
        )

    def reset(self):
        """将全部并行环境重置为完全相同的初始场景。"""
        self.pos[:] = torch.tensor(START_POS, device=DEVICE, dtype=self.pos.dtype)
        self.vel.zero_()
        self.end[:] = torch.tensor(END_POS, device=DEVICE, dtype=self.pos.dtype)
        self.obstacle[:, :2] = torch.tensor(OBSTACLE_POS, device=DEVICE, dtype=self.pos.dtype)
        self.obstacle[:, 2] = OBSTACLE_RADIUS
        self.done.zero_()
        self.time = 0.0
        return self.get_state(), self.done.clone()

    def step(self, action):
        """执行一步并返回 ``(next_obs, reward, done[N])``。"""

        active = ~self.done # 活着的环境
        previous_pos = self.pos.clone()

        active_agents = active[:, None, None]
        applied_action = torch.where(active_agents, action, torch.zeros_like(action))
        self.pos, self.vel = update_pos(self, applied_action, active)
        self.time += self.dec_step * self.sim_time

        distance_to_goal = torch.norm(self.pos - self.end, dim=-1)
        reached = torch.all(distance_to_goal <= self.end_radius, dim=-1)
        collision = self._segment_hits_obstacle(previous_pos, self.pos).any(dim=-1)
        timeout = torch.full_like(active, self.time >= self.total_time)
        just_done = active & (reached | collision | timeout)

        reward = compute_reward(self, active, reached, collision)
        self.done |= just_done
        return self.get_state(), reward, self.done.clone()

    def _segment_hits_obstacle(self, start, end):
        """检测一步内的线段是否穿过圆，避免大步长直接越过障碍物。"""
        center = self.obstacle[:, :2]
        segment = end - start
        length_sq = torch.sum(segment * segment, dim=-1)
        projection = torch.sum((center - start) * segment, dim=-1)
        projection = projection / length_sq.clamp_min(1e-8)
        projection = projection.clamp(0.0, 1.0).unsqueeze(-1)
        nearest = start + projection * segment
        return torch.norm(nearest - center, dim=-1) <= self.obstacle[:, 2]
