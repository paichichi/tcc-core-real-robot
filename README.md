# Real-Robot Policy Training and Evaluation

## 推荐的新实验：ViT、60 demos、纯视觉 absolute action

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
`checkpoint_050000.pt`，test 集只在选择完 checkpoint 后评估一次。新 checkpoint
发布到 Hugging Face 后，必须把新配置中的 `model_hub.revision` 固定为发布 commit，
再开始 shadow 或实机评估。

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

这个命令不连接机械臂。它读取 10 个实际训练 episode 的两路首帧，比较 policy
预测、记录 action 和首帧 state：

```bash
python scripts/eval_demo_first_frames.py \
  --backbone ours_rn50 \
  --demonstrations 80 \
  --task carrot \
  --episodes 10 \
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
