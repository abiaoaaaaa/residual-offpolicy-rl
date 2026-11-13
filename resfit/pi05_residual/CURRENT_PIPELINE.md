# Pi0.5 Residual RL：当前实现与轨迹隔离测试

## 运行结构

```text
VLAHost :8010
   │  GET /state（14关节 + 2夹爪 + 三路图像）
   ▼
collector（真机进程，10Hz）
   ├─ Pi0.5：约1Hz推理一次，输出30×16绝对关节目标
   │          取索引2,5,...,29，形成10个10Hz基础目标
   ├─ residual actor：三路128×128图像 + 64D state（raw 16D + context 48D） + base action
   │                  输出每关节最多±5°/100ms，夹爪residual=0
   ├─ final target = Pi绝对目标 + residual物理修正
   └─ 每个10Hz目标插值为3点，一次POST /action_chunk，source_hz=30
          VLAHost以300Hz生成31个setpoints，覆盖约100ms

learner（训练进程，gRPC :50051）
   ├─ online/offline双buffer
   ├─ 三套ImageNet ResNet18完全冻结（包括BatchNorm）
   ├─ prioritized replay，3-step，gamma=0.99
   ├─ online达到1000条后开始
   ├─ critic-only warmup 1000次
   ├─ 正式阶段每条online transition最多允许1个learner step
   └─ 每round更新critic 4次、actor 1次；episode边界发布/应用actor
```

## 核心配置

| 项目 | 当前值 |
|---|---:|
| Pi动作 | 30Hz绝对关节目标，30×16 |
| RL控制 | 10Hz |
| RL state | normalized raw 16D + previous actual residual 16D + actual absolute delta 16D + proposal/execution disagreement 16D |
| 三路相机 | front、left wrist、right wrist |
| Pi图像 | 224×224 |
| RL图像 | 128×128 |
| 视觉编码器 | 三套独立ImageNet ResNet18，完全冻结 |
| Residual | 前14关节±5°/100ms；夹爪关闭 |
| Low-pass | alpha=1.0（关闭） |
| Bridge限幅 | 每个30Hz waypoint最多0.08rad |
| VLAHost | HTTP 127.0.0.1:8010 |
| Learner | gRPC 127.0.0.1:50051 |
| Replay | capacity=200000，batch=256，PER α=0.6/β=0.4 |
| TD | n-step=3，gamma=0.99 |
| 更新 | UTD round=4，actor每round 1次 |
| 数据更新配额 | 默认每新增1条 online transition 最多1个正式 learner step |

键盘：`B` 开始RL回合，`A` 失败，`C` 成功，`Ctrl+1` 打开双夹爪、回零并等待3秒。
未按 `B` 时 Pi0.5 仍执行，但 residual 为零且不写 replay。

Replay 的 `action` 始终是 actor 提议的 normalized combined action；EMA、slew、
low-pass 和 bridge 后的实际 residual/absolute command 分别写入
`executed_residual_action`、`executed_absolute_action`，不能用执行动作覆盖 Q(s,a)。
离线 expert 转换器也按同样的 context 顺序和 reset/step 更新规则生成 64D state。

脚踏 VR 接管期间不结束当前 B→A/C 回合；VLAHost 停止 Pi chunk 下发，collector 仅在
`intervention.active/intervene_flag` 明确且 `executed_action_fresh=true` 时，用实际
`executed_action` 相对当前 Pi 参考关节角转换为归一化 action。完整 RL 回合进入 online；
这些已验证人工帧额外进入 learner 的 expert buffer。无法验证或超出 [-1,1] 的帧
不伪装成 Pi/actor 动作，并切断 n-step 连续性。部署前需重新构建并启动更新后的 VLAHost。

## 当前已知时序

最近真机日志中，普通chunk间隔平均约100.14ms；Pi0.5每10步同步重推理时平均约109.09ms，最大约129.03ms。

当前 bridge 始终以上一个已发送末端目标作为下一段插值起点，并在100ms动作周期末端采集 `next_obs`。Pi0.5在当前chunk第8步后后台预推理下一chunk，正常情况下切换无需额外等待。

## 隔离测试：只推理一次，然后无模型回放

### 1. 捕获当前观测下的 Pi0.5 输出

```bash
cd /home/xlb/code/residual-offpolicy-rl
export PYTHONPATH=/home/xlb/code/residual-offpolicy-rl:/home/xlb/data/code/code_new/lerobot_vlahost/src:/home/xlb/data/code/code_new/code_xlb/openpi/src

/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.capture_trajectory \
  --bridge-url http://127.0.0.1:8010 \
  --pi05-checkpoint /home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999 \
  --pi05-config pi05_zadai_ybs_joint_clean_951 \
  --task "Insert the tail of the yellow cable tie into its head to fasten it." \
  --output /home/xlb/data/pi05_residual_debug/current_pi05_chunk.json
```

JSON保存捕获state、原始30×16动作、抽样索引以及10×16回放目标，不保存图像像素。

### 2. 只检查起始姿态，不发送动作

```bash
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.replay_trajectory \
  --trajectory /home/xlb/data/pi05_residual_debug/current_pi05_chunk.json \
  --bridge-url http://127.0.0.1:8010
```

### 3. 真机回放一次

```bash
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.replay_trajectory \
  --trajectory /home/xlb/data/pi05_residual_debug/current_pi05_chunk.json \
  --bridge-url http://127.0.0.1:8010 \
  --execute
```

回放阶段不导入或加载Pi0.5、actor、critic。它只读取JSON中的10个绝对目标，按当前bridge逻辑逐个插值为3点并以10Hz提交。起始14关节与捕获姿态最大误差默认超过0.15rad时拒绝执行。结果写入同目录的 `current_pi05_chunk.replay_report.json`。

## 判断方式

- 回放仍明显抖：问题位于固定轨迹、bridge起点、限幅或VLAHost播放链路。
- 回放平稳、在线Pi抖：重点检查每秒重推理边界和相邻Pi chunk连续性。
- 只在每100ms边界抖：重点检查立即反馈被用作下一chunk插值起点。
- 只在每1秒边界抖：重点检查Pi0.5同步重推理和新旧30步chunk衔接。

## LeRobot完整回合回放

数据集 `/home/xlb/data/Data/lerobot/9_20_to_9_21_zadai_joint_clean` 是30Hz、16维绝对关节 action。默认选择动作变化最平滑的 episode 135（约26.8秒），每3帧组成一个 `source_hz=30` 的 `/action_chunk`。该测试不加载任何模型；`--execute` 才会下发真机动作。

```bash
cd /home/xlb/code/residual-offpolicy-rl
export PYTHONPATH=/home/xlb/code/residual-offpolicy-rl:/home/xlb/data/code/code_new/code_xlb/openpi/src

# 只检查起始姿态、数据形状和动作跳变
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.replay_lerobot_episode \
  --dataset /home/xlb/data/Data/lerobot/9_20_to_9_21_zadai_joint_clean \
  --episode 135 \
  --bridge-url http://127.0.0.1:8010

# 确认机器人姿态匹配后，再执行约26.8秒真机回放
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.replay_lerobot_episode \
  --dataset /home/xlb/data/Data/lerobot/9_20_to_9_21_zadai_joint_clean \
  --episode 135 \
  --bridge-url http://127.0.0.1:8010 \
  --execute
```
