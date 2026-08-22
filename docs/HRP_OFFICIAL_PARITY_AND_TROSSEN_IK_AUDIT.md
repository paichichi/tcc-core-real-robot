# HRP parity and Trossen joint-action audit

Audit date: 2026-08-22

## Final production decision

The policy learns the dataset's original seven-dimensional action without an
action-space conversion:

1. input state: six measured arm-joint positions plus measured gripper position;
2. target: six commanded arm-joint positions plus commanded gripper position
   from the original LeRobot `action` column;
3. runtime output: denormalized absolute joint/gripper position;
4. driver boundary: the existing bounded rollout calls `set_all_positions`, the
   same command used during collection and successful 359-frame demo replay.

The production policy does not use Cartesian pose conversion, differential
velocity, or inverse kinematics. The experimental Cartesian modules remain
isolated diagnostic code and are not referenced by the v8 config, image cache,
trainer, or runtime contract.

Every image buffer embeds a semantic manifest. Training fails before model setup
unless it explicitly identifies the original measured joint state and original
LeRobot joint-position action. This prevents accidental reuse of the previous
seven-dimensional Cartesian-velocity cache.

## HRP training parity

| Item | HRP release | This reproduction | Result |
|---|---|---|---|
| Policy | state token + visual token, token BatchNorm, MLP `[512, 512]`, ReLU, dropout `0.2` | same | parity |
| HRP/D4R ViT token | CLS token (`use_cls: true`) | raw MAE releases use CLS; TCC-trained ViT retains native patch mean | source-correct |
| Distribution | 5-mode GMM, minimum std `1e-4`, NLL | same | parity |
| Inference | categorical component with component std forced near zero | same | parity |
| Optimizer | Adam, lr `3e-4`, weight decay `1e-4` | same | parity |
| Batch/steps | 150 / 150,000 | same | parity |
| Holdout | fixed 500 transitions, seed `3904767649` | same | parity |
| Sampling | random transitions with replacement | same | parity |
| Train transform | HRP `medium` transform | same | parity |
| Eval transform | resize, no antialias, ImageNet normalization | same in training/runtime | parity |
| Precision | float32 | float32 | parity |
| Action | embodiment-specific | original Trossen joint-position goal | dataset/driver parity |
| Checkpoint | release includes optimizer/resume state | inference weights and statistics only | intentional: no resume |

HRP defines the visual/state policy and behavior-cloning recipe, not a universal
robot action space. Its examples are embodiment-specific. Preserving the action
that generated this dataset is therefore compatible with HRP while eliminating
an unnecessary robot-specific conversion.

Action normalization is fit only on training transitions. Its mean and standard
deviation are stored in each checkpoint. Inference reverses the normalization,
so the Driver receives values in the original joint/gripper position units.

## Real-environment adaptation

All 35,900 carrot frames (100 episodes x 359 frames) were audited:

- cached state and action match the source parquet bit-for-bit (`max diff = 0`);
- `action[t]` is closest to measured `state[t+2]`, consistent with the real
  controller following a nonblocking command roughly two 20 Hz frames later;
- training therefore keeps `action[t]` unchanged instead of converting it to a
  delta or shifting it to a different target;
- the rollout per-step envelope now uses maxima across all 100 demonstrations,
  rather than the narrower envelope from one replayed episode;
- target EMA is disabled (`alpha = 1`) because it added delay absent from data;
  the 0.3 s Driver interpolation, measured-position lead limiter, absolute data
  envelope, tracking checks, and E-stop gate remain active;
- measured data maxima set the runtime diagnostic ceilings to 2.8 rad/s for arm
  joints and 0.09 m/s for the gripper.

The trained policy uses only `cam_main`. RealSense evaluation now starts only
the serial-pinned D435 for this checkpoint. It does not make success depend on
the unused wrist D405 or include its acquisition latency. RGB8, 640x480 capture,
ImageNet normalization, and the HRP eval resize remain identical to training.

## Driver and safety boundary

The collection and rollout paths agree on:

- 20 Hz command cadence;
- nonblocking `set_all_positions` calls;
- 0.3 s movement goal (`6 / 20 Hz`);
- original dataset action envelope;
- per-frame delta, command-lead, and tracking-error limits.

IK is deliberately outside this path: an absolute joint-position action already
specifies the Driver goal, so applying IK would introduce a second, conflicting
action interpretation. Cartesian/IK tests remain useful for standalone robot
diagnostics only.

The driver path has been validated by demo replay, but a newly learned checkpoint
has not. Global `action_contract.enabled` remains false. Before a real learned
rollout, run shadow diagnostics and then a supervised bounded test from dataset
home with an operator and E-stop. Firmware 1.9.3 remains recommended because
Trossen documents abnormal-termination robustness fixes beyond 1.9.2.

## Frozen references

- HRP/Data4Robotics `hrp_release`:
  <https://github.com/SudeepDasari/data4robotics/tree/hrp_release>
- Trossen LeRobot collection fork:
  <https://github.com/Interbotix/lerobot/tree/trossen-ai>
- Trossen Arm Driver v1.9.3:
  <https://github.com/TrossenRobotics/trossen_arm/tree/v1.9.3>
- Dataset revision: `6ce893dc73fc6310ccbd46786f71a03a5b6a3da2`

Verification: Ruff passed, Python 3.10 CPU suite passed (110 tests), the real
two-episode cache retained 718/718 frames, and a one-iteration end-to-end RTX
5090 smoke completed with finite train/held-out GMM metrics. No physical robot
command was issued.
