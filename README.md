# Real-Robot Policy Training and Evaluation

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

10 帧 future delta 对应数据集 20 Hz 下的 0.5 秒目标。执行时使用 `0.1` gain
转换成一个 20 Hz 控制步，随后仍经过现有 driver 的逐步、累计、关节和 workspace
限制。

## 执行已发布的 policy

真实 policy 执行默认使用固定的 `ours_rn50`、60-demo policy、carrot 任务、两台
已配置的 RealSense、离线 Hugging Face 缓存和完整 359 步：

```bash
python scripts/run_policy.py --execute-policy --emergency-stop-ready
```

需要临时覆盖时可追加参数，例如 `--max-steps 30`、`--task pineapple` 或
`--online`。

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
  --demonstrations 60
```

下载完成后，可以使用离线模式再次检查：

```bash
python scripts/fetch_policy_assets.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
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
  --demonstrations 60 \
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
  --demonstrations 60 \
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
  --demonstrations 60 \
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
