# OpenPI Residual RL：AI 快速上下文

这份文件是给新对话 AI 的第一阅读入口。用户通常希望直接分析、修改、测试和给出真机命令；先读本文件，再读 `resfit/pi05_residual/PIPELINE.md` 和 `CURRENT_PIPELINE.md`。

## 项目目标

把 OpenPI 的 Pi0.5 当作冻结黑盒基策略，在真实双臂机器人上叠加高频 residual RL。当前任务是：

> Insert the tail of the yellow cable tie into its head to fasten it.

基策略来自 OpenPI native policy API，不使用 RLinf 作为运行框架；RL 网络和 replay 复用本仓库的 residual-offpolicy-rl 实现。

## 当前真实链路

```text
VLAHost :8010 /state
  14 个关节角 + 2 个夹爪值 + 拼接三视角图像
        ↓
Pi0.5 native infer：30 Hz、30×16、绝对关节目标
        ↓ 取索引 2,5,...,29
10 个 10 Hz Pi 参考目标
        ↓
Residual actor：三套冻结 ResNet18 + 可训练压缩/MLP
        ↓
base normalized delta + residual → [-1,1]
        ↓
Pi 绝对目标 + residual 物理修正
        ↓
10 Hz 每步插值成 3 个 30 Hz waypoint
        ↓
POST /action_chunk，VLAHost 播放并继续以末端目标为下一段锚点
```

重要动作语义：Pi0.5 输出是**绝对关节目标**，不是归一化 delta。当前 actor/replay `action` 保存 actor 提议的 normalized combined action；EMA/slew/bridge 后的实际 residual 与绝对命令单独保存，不能用滤波后的动作覆盖 critic 的 Q(s,a) 动作。Pi 参考目标之间的 10 Hz sequential delta 仅用于 residual contract，不能把 Pi 目标再当 delta 直接累加到实测状态。64D state 的后三段上下文在 collector 与离线转换器中完全一致。

## 当前配置

| 项目 | 当前值 |
|---|---:|
| Pi chunk | 30 个绝对目标，30 Hz |
| RL actor/replay | 10 Hz，每步 100 ms |
| 观测 | 三路图像 + 64 维 state（归一化 raw 16D + 48D 执行上下文） |
| Pi 图像 | 224×224 |
| RL 图像 | 128×128 |
| actor residual | 前 14 个关节最多 5°/100 ms；夹爪 residual=0 |
| residual filter | 默认 EMA `alpha=0.35` + 每个 100 ms 最多 1° residual slew |
| action low-pass | 绝对目标可选 `alpha`；默认 `1`，residual 自身仍受滤波/限速 |
| bridge 限幅 | 每个 30 Hz arm waypoint 最多 0.08 rad |
| replay TD | n-step=3，gamma=0.99，PER |
| learner | critic warmup=1000，正式 UTD=4（4 critic + 1 actor） |
| 更新配额 | 默认每新增 1 条 online transition 最多 1 个 learner step |
| 混合采样 | `offline_fraction=0.5` |

三套 ResNet18 完全冻结（含 BatchNorm）；Actor 的视觉压缩层、Actor MLP、Critic spatial embedding/Q heads 训练。Actor 最终 tanh 输出 `[-1,1]`。

## 回合、按键和干预

- 未按 `B`：Pi0.5 继续稳定执行，residual 为零，不写 RL replay。
- `B`：开始一个 RL 回合。
- `A` 或 `Q`：失败结束回合。
- `C`：成功结束回合。
- `Ctrl+1`：停止干预会话，go home，打开双夹爪，等待 3 秒，再恢复 Pi 控制。
- VR/脚踏接管：不结束 B→A/C 回合；策略暂停发 chunk。

完整 B→A/C 回合进入 online buffer。只有 VLAHost 明确报告 `active/intervene_flag/source=human/phase=vr`、两侧实际 mux 输出足够新鲜（`executed_action_fresh=true`），且实际绝对动作可无裁剪转换到 critic `[-1,1]`，该帧才额外复制到 expert buffer。不能用未执行的 `scaled_action` 冒充人工动作。

人工帧之间的间隙会切断 expert 的 n-step 连续性；无法验证或超出归一化范围的帧会跳过，而不是静默裁剪。

## 常用路径

```text
项目：        /home/xlb/code/residual-offpolicy-rl
OpenPI 源码： /home/xlb/data/code/code_new/code_xlb/openpi/src
兼容 LeRobot：/home/xlb/data/code/code_new/lerobot_vlahost/src
Pi checkpoint：/home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999
训练输出：    /home/xlb/data/pi05_residual_v8_clean951
online replay：.../pi05_residual_v8_clean951/online_replay
offline expert：.../pi05_residual_v8_clean951/offline_expert_38
VLAHost：     http://127.0.0.1:8010
键盘：        /dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd
```

## 标准启动

先确保 VLAHost 使用当前源码重新构建并启动；旧服务没有 `executed_action_fresh`，collector 会拒绝干预训练：

```bash
cd /home/xlb/data/code/code_new/Apex_Deploy
source /opt/ros/humble/setup.bash
colcon build --packages-select vlahost
source install/setup.bash
ros2 launch vlahost vlahost_server.launch.py host:=0.0.0.0 port:=8010 end_effector:=gripper
```

验证接口：

```bash
curl -s http://127.0.0.1:8010/state | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["intervention"].get("executed_action_fresh", "MISSING"))'
```

训练 learner：

```bash
cd /home/xlb/code/residual-offpolicy-rl
export PYTHONPATH=/home/xlb/code/residual-offpolicy-rl:/home/xlb/data/code/code_new/lerobot_vlahost/src:/home/xlb/data/code/code_new/code_xlb/openpi/src
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python -m resfit.pi05_residual.train \
  --pi05-checkpoint /home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999 \
  --pi05-config pi05_zadai_ybs_joint_clean_951 \
  --task "Insert the tail of the yellow cable tie into its head to fasten it." \
  --replay-dir /home/xlb/data/pi05_residual_v8_clean951/online_replay \
  --checkpoint /home/xlb/data/pi05_residual_v7_clean951/learner.pt \
  --residual-checkpoint /home/xlb/data/pi05_residual_v7_clean951/actor.pt \
  --chunk-size 30 --rl-chunk-size 10 --residual-joint-deg 5 \
  --residual-ema-alpha 0.35 --residual-slew-limit-deg 1.0 \
  --action-lowpass-alpha 1 --bridge-joint-step-limit-rad 0.08 \
  --learning-starts 1000 --critic-warmup-steps 1000 \
  --batch-size 256 --updates-per-round 4 --actor-updates-per-round 1 \
  --max-learner-steps-per-online-transition 1 --n-step 3 \
  --sampling-strategy prioritized_replay \
  --offline-replay-dir /home/xlb/data/pi05_residual_v8_clean951/offline_expert_38 \
  --offline-fraction 0.5 --async-grpc
```

collector：

```bash
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python -m resfit.pi05_residual.collect \
  --bridge-url http://127.0.0.1:8010 \
  --pi05-checkpoint /home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999 \
  --pi05-config pi05_zadai_ybs_joint_clean_951 \
  --task "Insert the tail of the yellow cable tie into its head to fasten it." \
  --output-dir /home/xlb/data/pi05_residual_v8_clean951/online_replay \
  --keyboard-device /dev/input/by-id/usb-HS6209_A4tech_2.4G_Wireless_Device-event-kbd \
  --chunk-size 30 --rl-chunk-size 10 --residual-joint-deg 5 \
  --residual-ema-alpha 0.35 --residual-slew-limit-deg 1.0 \
  --action-lowpass-alpha 1 --bridge-joint-step-limit-rad 0.08 \
  --pi-session-minutes 30 --reset-sleep-seconds 3 \
  --async-grpc --control-hz 10
```

## 安全调试顺序

不要一上来真机训练。优先按以下顺序：

1. `capture_trajectory` 捕获 Pi chunk；不发送动作。
2. `replay_trajectory` 不加 `--execute` 检查起始姿态和跳变。
3. 同一轨迹加 `--execute`，确认固定轨迹和 bridge 平滑。
4. `pi_only` 连续推理，确认 Pi chunk 边界稳定。
5. `residual_shadow` 只运行 actor 但发送零 residual。
6. 最后才启动 learner + collector；先按 B，短回合按 A/C。

三路轨迹回放和 capture 命令见 `resfit/pi05_residual/CURRENT_PIPELINE.md`。

## 常见诊断

- `FileNotFoundError: /path/to/offline.npz`：误用了占位路径；改用 `--offline-replay-dir` 的真实目录。
- `No module named lerobot.transport`：PYTHONPATH 缺少 `lerobot_vlahost/src`。
- VLAHost connection refused：先启动 8010 服务。
- `waiting for online replay`：online transition 尚未达到 `learning-starts`，不是模型崩溃。
- actor 输出趋近固定偏置：检查是否长时间由人工接管、expert action 远超 5° actor 能力；不要把人工成功直接算作自主成功。
- `learner.pt` 步数低于 `actor.pt`：正常，actor 发布频率和 learner checkpoint 保存频率不同；停止 learner 时用 Ctrl+C 等它保存。
- 真机抖动：先比较固定轨迹 replay、Pi-only、residual shadow，再查推理延迟、chunk 边界和 bridge 锚点；不要先改 reward。

## 关键文件

- `resfit/pi05_residual/collect.py`：真机 10 Hz collector、键盘、VR 接管和 replay 写入。
- `resfit/pi05_residual/train.py`：online/expert 双 buffer、warmup、UTD、PER 和 checkpoint。
- `resfit/pi05_residual/env.py`：Pi absolute target、normalized base delta、residual 转换。
- `resfit/pi05_residual/vlahost.py`：HTTP bridge、`/action_chunk` 三 waypoint、实际人工动作校验。
- `resfit/pi05_residual/action_codec.py`：绝对目标到 sequential normalized delta。
- `/home/xlb/data/code/code_new/Apex_Deploy/robot_node/vlahost/vlahost/vla_node.py`：ROS/VLAHost 服务端和 intervention snapshot。

修改代码后至少运行：

```bash
cd /home/xlb/code/residual-offpolicy-rl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/home/xlb/code/residual-offpolicy-rl:/home/xlb/data/code/code_new/lerobot_vlahost/src:/home/xlb/data/code/code_new/code_xlb/openpi/src \
  /home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python -m pytest -q \
  tests/test_pi05_bridge_chunk.py tests/test_pi05_residual_codec.py tests/test_pi05_residual_runtime.py
```

## 给新 AI 的工作约束

先确认用户要的是分析、诊断还是改代码；涉及真机时只做只读检查和模拟测试，除非用户明确要求下发动作。不要把旧版 ACT、旧版 `/action`、旧版 30 Hz checkpoint 与当前 absolute-target + `/action_chunk` 合同混用。修改前检查当前 dirty worktree，保留用户已有训练结果和 replay。
