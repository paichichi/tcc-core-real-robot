# Real-robot safety gate

No motion-producing program belongs in the initial repository revision.

Before adding or running actuation code, the on-site operator must verify:

- the work area is clear and the robot is mechanically unobstructed;
- the emergency stop is reachable and tested;
- the robot model, follower role, controller IP, driver version, and firmware
  version match the installed hardware;
- camera names are mapped to their physical locations;
- action semantics and units are confirmed from both documentation and data;
- joint, velocity, gripper, and Cartesian workspace limits are explicit;
- execution defaults to dry-run and requires an explicit opt-in flag;
- the first hardware test is limited, slow, supervised, and independently
  recoverable.

Never commit credentials, access tokens, local IP overrides, private datasets,
videos, or model checkpoints.
