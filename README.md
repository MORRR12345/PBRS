# PPO 二维避障：Hard / Soft / PBRS

固定起点到固定终点，中间有一个圆形障碍物。使用同一个 PPO 配置分别训练三种避碰奖励，并比较累计成功率、episode return 与损失。训练日志保留逐 episode 的回报与成功标记、逐次 PPO 更新的 loss；绘图时仅对 return 和 loss 做滑动平均。Python 3.10+，无需 pygame。

## 文件

| 文件 | 用途 |
| --- | --- |
| `config.py` | 场景、奖励及训练参数 |
| `env.py` | Gymnasium 连续动作环境 |
| `reward.py` | 奖励、净空距离和 PBRS 势能计算 |
| `train.py` | 依次训练并保存模型、episode 日志 |
| `plot_results.py` | 生成三张训练对比图 |
| `play.py` | 确定性回放并保存路径图 |
| `requirements.txt` | Python 依赖 |

## 环境与奖励

绘图参考范围为 `[0, 10] × [0, 10]`，智能体可自由越过该范围；起点 `(1, 5)`，目标 `(9, 5)`，圆障碍物中心 `(5, 5)`、半径 `1.1`。智能体半径 `0.15`，每步动作是 `[-1, 1]^2`，位移为 `0.2 × action`。观测为 8 维 `float32`：当前位置、目标相对向量、障碍物相对向量、目标距离、障碍物净空距离。净空距离 `d_obs = ||position - center|| - obstacle_radius - agent_radius`，小于等于零视为碰撞。

公共奖励为每步 `10 × (上一步目标距离 - 当前目标距离) - 0.1`，成功额外 `+20`，处于障碍物内的每步额外 `-10`。碰撞后可以继续移动；只有到达目标才终止。达到 100 步仍未成功则截断。目标距离不大于 `0.2` 且未碰撞为成功。各方案额外奖励：

- `hard`：无额外奖励。
- `soft`：每步加 `-w × max(0, d_safe - d_obs)²`。
- `pbrs`：设 `Φ(s) = -w × max(0, d_safe - d_obs)²`，每步加 `γΦ(s') - Φ(s)`。

默认 `w=1`、`d_safe=1.5`、`γ=0.99`，均可在 `config.py` 调整。奖励分项由 `reward.py` 计算，并通过环境的 `info["reward_parts"]` 返回。PBRS 与 PPO 使用同一个 `γ`；到达目标时下一状态势能置零，超时时作为采样截断继续估计末状态价值。三种奖励的 return 数值尺度不同，比较避障效果时应主要看成功率与路径。

## 运行

在项目根目录运行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py --episodes 100
python plot_results.py
python plot_results.py --window 50 --loss-window 5  # 可选：调整平滑窗口
python play.py --all
# 或只看一种：python play.py --reward pbrs
```

训练时终端会依次显示 hard、soft、pbrs 三条进度条。

每轮让 `NUM_ENVS` 个环境各完成一局，将这些完整轨迹合并后进行一次 PPO 更新。`--episodes` 控制每种方案的采样轮数和更新次数，总局数为 `--episodes × NUM_ENVS`。每轮记录各环境的平均回报、成功率、碰撞步数、超时率及长度，并记录一组 actor/critic loss。`success_rate.png` 显示累计平均成功率；return 默认按 30 轮平滑，loss 默认按 5 次更新平滑。

每轮同时对活动环境批量推理；某个环境完成一局后即停止采样，等待其余环境结束。完整轨迹按环境拼接，保留局间边界，再交给 SB3 的 PPO 优化器。`TOTAL_EPISODES` 是每种奖励方案的轮数，`NUM_ENVS` 是每轮的环境数。

训练保存 `models/ppo_{hard,soft,pbrs}.zip` 和 `logs/{hard,soft,pbrs}.npz`；绘图保存 `figures/success_rate.png`、`episode_return.png`、`losses.png`、`play_paths.png`。三次训练使用相同随机种子和超参数，固定起点与环境，结果仍受优化随机性影响。actor / critic loss 从 SB3 logger 在每次更新后读取。

日志另存每轮的平均 `length`、`collision`（处于障碍物内的步数）和 `timeout` 率，可用来排查失败原因。旧模型和旧日志对应修改前的环境与奖励，修改后需重新执行训练、绘图和回放。
