# Real-Robot Policy Training and Evaluation

## 当前实验：V10 shared RN50 + main policy + gated wrist residual

当前配置为
`configs/experiment_v10_shared_rn50_main_wrist_residual_100.yaml`。V10 只测试
`ours_rn50`，两路相机共享同一个 RN50 并端到端微调；预训练 BatchNorm running
statistics 保持冻结。`cam_main` 独立预测完整 7 维动作，`cam_wrist` 只产生有界动作
修正：

```text
action = main_action + sigmoid(gate) * 0.25 * tanh(wrist_correction)
```

main 和 wrist 使用独立 projection head，但共享视觉 backbone。gate 初始 bias 为
`-2.0`，训练时以 `0.2` 概率屏蔽 wrist，使主视角始终能够单独完成全局定位和运动，
腕部视角只补充接近、对准和抓取阶段的局部信息。

训练图像在送入共享 RN50 前加入 train-only 光度增强：brightness/contrast
各 `0.15`、saturation `0.10`、hue `0.02`，以及 `0.10` 概率的轻微 Gaussian
blur。两路相机独立采样增强，以覆盖 D435/D405 不同的颜色与曝光响应。validation、
test 和实机 eval 仍使用完全确定性的 resize + ImageNet normalization。V10 不使用
flip、rotation 或 random crop，避免改变图像坐标却保留原 absolute-action label。

训练目标就是数据集原始 `action[t]`：6 维关节绝对目标（rad）加 1 维夹爪绝对目标
（m）。只使用 train episode 的逐维 action mean/std 做 normalization；runtime
denormalize 后直接交给现有 Trossen joint-position 安全层和
`set_all_positions(target, goal_time, False)`。V10 不使用 delta adapter、Cartesian
velocity 或 IK。缓存、checkpoint 和 runtime 都强制声明 action 相对 measured state
领先两帧；20 Hz 下两帧严格对应 driver 的 `0.1 s` 非阻塞插值。

100 个 demo 按完整 episode 固定划分为 90/10 train/test，避免帧泄漏。V10 不使用
validation 或 early stopping：训练固定为 50K steps，部署固定使用最后的
`checkpoint_050000.pt`，test 只在训练结束后评估一次，不能根据 test 指标选择
checkpoint。batch size 为 32；policy head、projection/gate、backbone 的 learning
rate 分别为 `1e-3`、`1e-4`、`1e-5`。损失由 fused action MSE、main-only
auxiliary MSE 和 wrist residual regularization 组成。每 5K 步保存恢复用 checkpoint，
但不在它们之间选择“最佳”模型。每个 5K 窗口同时保存平均 loss、逐维 action
MAE、gate、residual 和 gradient norm；这些指标只用于判断收敛，不参与模型选择。

完整训练和本地 checkpoint 离线闸门命令放在 `LINUX_COMMANDS.txt`。首帧诊断默认
只检查 10 个 held-out test episode，比较预测绝对目标与 demo 记录目标；超过记录首步 envelope 的模型会得到
`Decision: BLOCKED`，不应进入实机 rollout。

## 历史实验：V9 independent encoders 与 V6 gated features

## 上一版实验：v3 proprioception + absolute action

v2 纯视觉单帧 MLP 在 20 Hz 实机 rollout 中约一半相邻输出发生方向反转，359
帧中 308 帧触发 limiter，并在高处形成错误固定点。v3 使用
`configs/experiment_proprio_absolute_60.yaml`：冻结双摄像头 backbone，拼接经过
train split 统计量归一化的 7-D 当前机器人状态，直接预测 7-D absolute action。
它不使用旧 v1 的 future-delta。训练仍采用 `[256, 256]` MLP、SmoothL1 和
60/20/20 episode split。

训练命令已集中写入 `LINUX_COMMANDS.txt`。真实执行端还会从实测 home 初始化
`alpha=0.25` 的 policy-target EMA，再进入原有 dataset envelope、逐帧 slew、command
lead 和速度限制。报告同时保留 `raw_action` 与过滤后的 `action`，首帧安全检查仍使用
raw policy，EMA 不会掩盖不安全 checkpoint。

ViT 和 RN50 v3 policy 已固定到 Hugging Face commit
`47e78ae1595539ef80aa5c06c6ad3f75659ec33e`。runtime contract 检查会拒绝把旧 v2
checkpoint 当成 v3。

## 已完成的 v2 实验：ViT、60 demos、纯视觉 absolute action

连续两次 RN50 rollout 和一次 ViT rollout 都在 driver 正常跟踪的情况下走向
近似固定的错误轨迹。新实验与旧 v1 隔离，使用
`configs/experiment_visual_absolute_60.yaml`：每个任务固定划分 60 个 train、
20 个 validation、20 个 test episode；冻结 `ours_vit`，MLP 只输入两路视觉特征
和 task ID，不输入 proprioception，直接预测与 dataset replay 同语义的 7 维
absolute action。

必须使用新的 cache root，避免旧 80/10/10 cache 中残留的 episode 混入：

```bash
python scripts/cache_policy_features.py --config configs/experiment_visual_absolute_60.yaml --hub-backbone ours_vit --dataset-root datasets/pick_and_place_4_object_diverse --tcc-source-root /home/robotarm/TCC-core --cache-root runs/feature_cache_v2/ours_vit/60 --device cuda:0
```

缓存完成后训练：

```bash
python scripts/train_policy.py --config configs/experiment_visual_absolute_60.yaml --cache-root runs/feature_cache_v2/ours_vit/60 --output-dir runs/tcc_mlp_bc_v2/ours_vit/60 --device cuda:0
```

训练脚本会拒绝 split 与配置不一致的 cache。部署前使用 validation 最优的
`checkpoint_050000.pt`，test 集只在选择完 checkpoint 后评估一次。已发布的
60-demo ViT policy 固定到 Hugging Face commit
`1b104dfdd7b41d9619c0b128ccf321e77a03469a`；先下载校验并执行 shadow，不直接进入
实机 rollout。

当前代码兼容旧的 `tcc_mlp_bc_v0` checkpoint，并提供改进后的
`tcc_mlp_bc_v1_future_delta`：冻结视觉 backbone，输入两路视觉特征、7 维当前
机器人状态和任务 ID，预测 10 帧后的状态增量。训练使用按 episode 划分的
80/10/10 train/validation/test、LayerNorm、SmoothL1，并按 validation loss 选择
部署 checkpoint。

## 训练改进版 MLP policy

先为指定的冻结 backbone 缓存两路图像特征：

```bash
python scripts/cache_policy_features.py --hub-backbone ours_rn50 --dataset-root datasets/pick_and_place_4_object_diverse --tcc-source-root /home/robotarm/TCC-core --cache-root runs/feature_cache/ours_rn50 --device cuda:0
```

再训练 policy head：

```bash
python scripts/train_policy.py --cache-root runs/feature_cache/ours_rn50 --output-dir runs/tcc_mlp_bc_v1/ours_rn50/80 --device cuda:0
```

输出的 `checkpoint_050000.pt` 和 `checkpoint_best.pt` 都是 validation 最优权重；
前者兼容现有 Hugging Face 路径。`checkpoint_last.pt` 是第 50,000 步权重，
`metrics.json` 记录训练曲线、最优 validation 指标和只评估一次的 test 指标。

10 帧 future delta 对应数据集 20 Hz 下的 0.5 秒目标。官方 driver 使用 `0.3 s`
非阻塞插值，因此执行时使用 `0.3 / 0.5 = 0.6` gain，使预测目标和 controller
时间尺度一致；实时 runner 会显式覆盖旧 HF checkpoint 中保存的 `0.1`。指令随后
仍经过现有 driver 的逐步、累计、关节和 workspace 限制。逐关节单步上限取自
成功执行的 episode 33 replay 实测最大值，而不是统一的
`0.02 rad`；因此 policy 控制路径允许复现 replay 的运动时间尺度，同时运行时仍限制
机械臂速度不超过 `1.5 rad/s`、夹爪速度不超过 `0.06 m/s`。

## 执行已发布的 policy

真实 policy 执行默认使用固定的 `ours_rn50`、80-demo policy、carrot 任务、两台
已配置的 RealSense、离线 Hugging Face 缓存和完整 359 步：

```bash
python scripts/run_policy.py --execute-policy --emergency-stop-ready
```

需要临时覆盖时可追加参数，例如 `--max-steps 30`、`--task pineapple` 或
`--online`。

数据集标准 rollout 长度仍为 359 步。成功 replay 的逐帧最大关节速度为
`[0.504, 0.824, 0.923, 1.228, 0.732, 1.205] rad/s`；policy limiter 使用同一条
replay 的逐关节最大 step，并根据官方 `0.3 s` 非阻塞插值保留六帧 command lead。
为了诊断末段行为，真实 clipped rollout
允许显式扩展到最多 900 步；动作范围、单步变化、command lead、workspace 和
tracking 限制不会解除：

```bash
python scripts/run_policy.py --execute-policy --emergency-stop-ready --max-steps 900
```

默认使用 `--camera-read-mode latest`：两台 30 FPS 相机在后台持续采集同步帧，
20 Hz policy 循环读取最新帧，使相机等待与 GPU 推理重叠。需要对照旧路径时可追加
`--camera-read-mode synchronous`。报告会分别记录 camera、robot state、policy 和
command 的 median/p95 延迟。

## 1. 进入项目并激活环境

```bash
cd /home/robotarm/tcc-core-real-robot
source .venv/bin/activate
```

## 2. 安装 eval 依赖

```bash
python -m pip install -e '.[train,eval,robot,dev]'
```

## 3. 确认两个 RealSense 相机

```bash
python scripts/inspect_realsense_sdk.py
```

正式 eval 通过 RealSense SDK 按设备序列号读取明确的 `color/RGB8` 流，不再依赖
可能在重启后变化、并且可能指向深度流的 `/dev/video*` 编号：

```text
cam_main  = D435, serial 838212073584
cam_wrist = D405, serial 409122274608
stream    = color, RGB8, 640x480 @ 30 FPS
```

首次运行或重新插拔相机后，先执行纯相机采集：

```bash
python scripts/capture_policy_frames.py \
  --cam-main-serial 838212073584 \
  --cam-wrist-serial 409122274608
```

确认输出图片颜色正常后再运行 policy。相机底层固定为 30 FPS，policy 按 dataset
metadata 以 20 Hz 读取 observation；代码保留 warmup、读取重试、最大帧对时间差和
flat-frame 检查。V4L2 仅作为显式指定的兼容后端，不用于默认 eval。

## 4. 下载并校验模型

```bash
python scripts/fetch_policy_assets.py \
  --backbone ours_rn50 \
  --demonstrations 80
```

下载完成后，可以使用离线模式再次检查：

```bash
python scripts/fetch_policy_assets.py \
  --backbone ours_rn50 \
  --demonstrations 80 \
  --offline
```

## 5. 找到 TCC-Core 源码目录

```bash
find /home/robotarm -path '*/xirl/models.py' -print
```

例如，如果结果是：

```text
/home/robotarm/TCC-core/xirl/models.py
```

那么 `--tcc-source-root` 应填写 `/home/robotarm/TCC-core`。

## 6. 先运行 demo 首帧离线诊断

这个命令不连接机械臂。它读取 10 个 held-out test episode 的两路首帧，比较 policy
预测、记录 action 和首帧 state：

```bash
python scripts/eval_demo_first_frames.py \
  --backbone ours_rn50 \
  --demonstrations 80 \
  --task carrot \
  --episodes 10 \
  --split test \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto
```

结果保存到 `outputs/demo_first_frames_*.txt`。如果结果为 `BLOCKED`，先检查
policy 训练和输入预处理，不要继续真实动作执行。

## 7. 运行 10 步 shadow eval

如果 TCC-Core 位于 `/home/robotarm/TCC-core`，可以直接运行：

```bash
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 80 \
  --task carrot \
  --camera-backend realsense-sdk \
  --cam-main-serial 838212073584 \
  --cam-wrist-serial 409122274608 \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto \
  --execute-home \
  --max-steps 10
```

## 8. 检查输出

```bash
ls -lt outputs/policy_shadow_*.txt | head
```

```bash
less "$(ls -t outputs/policy_shadow_*.txt | head -n 1)"
```

报告末尾应出现：

```text
Decision: PASS
```

同时检查：

- `Completed steps: 10/10`
- `home_staging_completed: PASS`
- `first_arm_delta_safe: PASS`
- `first_gripper_delta_safe: PASS`
- 两个相机分辨率没有报错
- 每一步都输出 7 维有限数值
- `inference_ms` 和 `Observed rate` 满足实时运行需求

## 9. 运行完整 359 步 shadow eval

10 步测试通过后，将 `--max-steps` 改成 359：

```bash
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 80 \
  --task carrot \
  --camera-backend realsense-sdk \
  --cam-main-serial 838212073584 \
  --cam-wrist-serial 409122274608 \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto \
  --execute-home \
  --max-steps 359
```

可用任务：

```text
carrot
pineapple
starfruit
strawberry
```

## 安全状态

`--execute-home` 只执行回 home；`--execute-policy` 才会启用经过裁剪的真实 rollout，
且必须同时给出 `--emergency-stop-ready`。不要使用保留参数 `--execute`。程序退出时
会调用官方 driver cleanup 并恢复 Idle。新版 policy 不绕过任何现有动作边界。

持续人工监督模式使用 `--run-until-stopped`，且不能同时指定 `--max-steps`。该模式
没有 policy 步数终点，按 `q` 后完成最后目标校验并恢复 Idle；`Ctrl-C` 也会触发
driver cleanup。默认仍有 300 秒 wall-clock watchdog，可通过
`--watchdog-seconds` 显式设置。动作 envelope、逐步限幅、command lead、速度检查、
相机 watchdog 和官方 driver cleanup 均保持启用。当前 V10 没有 success head，因此
这个模式是“运行到人工停止”，不是“自动判断任务完成”。
