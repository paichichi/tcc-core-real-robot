# Dataset contract

Source: `UoA-Trossen-Arm/pick_and_place_4_object_diverse` on Hugging Face.

The repository contains four object-specific pick-and-place tasks with 100
episodes per task. Each task reports 35,900 frames and 200 videos, stored as
Parquet data and AV1 MP4 video streams.

## Recorded fields

- `observation.state`: float32, shape `(7,)`
- `observation.cartesian_position`: float32, shape `(6,)`
- `action`: float32, shape `(7,)`
- `observation.images.cam_main`: RGB, 640 x 480, 20 FPS
- `observation.images.cam_wrist`: RGB, 640 x 480, 20 FPS

## Open questions to resolve before training

1. Verify which physical camera maps to `cam_main` and `cam_wrist`.
2. Verify whether `action` is an absolute joint target, delta, or another command.
3. Confirm the seventh action/state component and its units.
4. Check timestamp synchronization and missing/corrupt frames.
5. Pin the exact dataset revision used for every experiment.
