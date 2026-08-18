# Real-Robot Policy Evaluation

当前只运行 shadow evaluation：读取两个相机并输出 policy 预测动作，但不会连接或驱动机械臂。

## 1. 进入项目并激活环境

```bash
cd /home/robotarm/tcc-core-real-robot
source .venv/bin/activate
```

## 2. 安装 eval 依赖

```bash
python -m pip install -e '.[eval,dev]'
```

## 3. 确认两个相机的 Linux 路径

```bash
ls -l /dev/v4l/by-id/
```

记录主相机 `cam_main` 和腕部相机 `cam_wrist` 对应的路径。

如果 `/dev/v4l/by-id/` 不存在，可以检查：

```bash
ls -l /dev/video*
```

直接抓取所有视频节点并显示带编号的画面：

```bash
python scripts/preview_cameras.py
```

在弹出的总览中人工判断哪一个画面是主相机、哪一个是腕部相机。按 `q`
或 `Esc` 关闭窗口。总览同时保存到 `outputs/camera_probe_<timestamp>.jpg`。

如果驱动电脑没有图形界面：

```bash
python scripts/preview_cameras.py --no-display
```

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

## 6. 运行 10 步 shadow eval

先替换主相机、腕部相机和 TCC-Core 路径，然后运行：

```bash
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
  --task carrot \
  --cam-main /dev/v4l/by-id/MAIN_CAMERA \
  --cam-wrist /dev/v4l/by-id/WRIST_CAMERA \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto \
  --max-steps 10
```

如果只知道相机编号，可以临时使用：

```bash
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
  --task carrot \
  --cam-main 0 \
  --cam-wrist 2 \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto \
  --max-steps 10
```

## 7. 检查输出

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
- 两个相机分辨率没有报错
- 每一步都输出 7 维有限数值
- `inference_ms` 和 `Observed rate` 满足实时运行需求

## 8. 运行完整 359 步 shadow eval

10 步测试通过后，将 `--max-steps` 改成 359：

```bash
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
  --task carrot \
  --cam-main /dev/v4l/by-id/MAIN_CAMERA \
  --cam-wrist /dev/v4l/by-id/WRIST_CAMERA \
  --tcc-source-root /home/robotarm/TCC-core \
  --offline \
  --device auto \
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

当前 `run_policy.py` 是 shadow-only。不要添加 `--execute`；该参数会被安全检查主动拒绝。预测动作只会写入 TXT 报告，不会发送给 Trossen 机械臂。
