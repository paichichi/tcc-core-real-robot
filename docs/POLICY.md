# TCC-MLP-BC v0

The first policy baseline follows the small language-conditioned behavior
cloning heads used by R3M, LIV, DecisionNCE, and AcTOL. It is intentionally a
reactive, single-step policy:

```text
cam_main  -> shared frozen TCC backbone -> feature_main  --+
cam_wrist -> shared frozen TCC backbone -> feature_wrist --+-> MLP -> 7-D action
four-way task ID                         -> one-hot -------+
```

The MLP has two 256-unit ReLU layers and is optimized with Adam for 50,000
gradient steps using a batch size of 32 and learning rate of `1e-3`. Actions
are standardized with statistics computed from training episodes only. The
first baseline deliberately excludes proprioception so it matches the AcTOL
real-robot configuration. A state-conditioned ablation can follow after this
baseline is established.

`evaluation.max_rollout_steps: 359` is a timeout matching the recorded episode
length. It is not an action chunk. The policy always predicts exactly one
action.

## Episode split

Each task contributes 60 deterministically selected training episodes, for 240
demonstrations in total. There is no validation or offline test split. The
remaining 40 episodes per task are unused. We always use the final 50,000-step
checkpoint and evaluate it through real-robot rollouts.

## Cache frozen features

## Fetch pinned Hub assets on Linux

The model repository and full commit revision are pinned in
`configs/experiment.yaml`. Download and verify one matched backbone-policy pair:

```bash
python scripts/fetch_policy_assets.py \
  --backbone ours_rn50 \
  --demonstrations 60
```

After the first successful download, the same command can run without network
access by adding `--offline`. Select `ours_vit` or any other configured
backbone with `--backbone`. The command verifies the backbone size and SHA256
from the repository manifest and prints the immutable local cache paths.

Feature caching also resolves the configured Hub backbone automatically. A
local checkpoint can still be supplied explicitly with `--checkpoint`.

Install the training dependencies and set `backbone.tcc_source_root` in
`configs/experiment.yaml`, or pass the TCC-Core source path on the command
line. The frozen checkpoint is downloaded from the pinned Hub revision by
default:

```bash
python -m pip install -e '.[train,dev]'
python scripts/cache_policy_features.py \
  --tcc-source-root /path/to/TCC-core \
  --dataset-root /path/to/pick_and_place_4_object_diverse \
  --cache-root runs/tcc_features
```

The cache stores backbone features, actions, states, and task indices. It does
not actuate the robot. Existing episode shards are reused unless `--overwrite`
is given.

## Shadow evaluation

Before using live cameras, evaluate recorded training-demo first frames:

```bash
python scripts/eval_demo_first_frames.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
  --task carrot \
  --episodes 10 \
  --tcc-source-root /path/to/TCC-core \
  --offline
```

This separates a policy/checkpoint failure from live camera domain shift by
comparing each prediction with the recorded first-frame action and state. It
never imports the robot driver.

Then identify the two stable Linux camera device paths and run the trained
checkpoint in shadow mode:

```bash
python -m pip install -e '.[train,eval,robot,dev]'
python scripts/run_policy.py \
  --backbone ours_rn50 \
  --demonstrations 60 \
  --task carrot \
  --cam-main /dev/v4l/by-id/MAIN_CAMERA \
  --cam-wrist /dev/v4l/by-id/WRIST_CAMERA \
  --tcc-source-root /path/to/TCC-core \
  --offline \
  --execute-home \
  --max-steps 10
```

The command restores the policy architecture and action normalization from the
50,000-step checkpoint. It applies the same RGB resize and ImageNet
normalization used during feature caching, predicts denormalized 7-D absolute
actions, and writes a human-readable report under `outputs/`. Use the full 359
steps only after the 10-step camera and latency smoke test passes.

`run_policy.py` keeps policy predictions shadow-only. `--execute-home` is a
separate, explicit staging action: it moves slowly to the median first state of
the 400 demonstrations, holds that pose, checks the first predicted action
against the measured state, then restores Idle. Passing `--execute` still
fails closed; shadow output is never sent to the Trossen driver.

## Train

```bash
python scripts/train_policy.py \
  --cache-root runs/tcc_features \
  --output-dir runs/tcc_mlp_bc_v0
```

The output contains `checkpoint_050000.pt` and training diagnostics. Training
error is not a robot-policy evaluation or permission to execute on hardware.
Before deployment, the 7-D action convention, units, limits, and watchdog
behavior must be verified independently.
