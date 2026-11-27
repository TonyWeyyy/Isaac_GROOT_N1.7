# Finetuning & Deploying GR00T N1.7 on Seeed reBot Arm B601-RS (3 Cameras)

This directory contains the modality configuration, hardware-independent safety/control utilities, and closed-loop real-robot evaluation client for the **Seeed reBot Arm B601-RS** (RobStride motors over SocketCAN) with a 3-camera setup:

- `wrist`: Orbbec Gemini 2 RGB (bound by serial number)
- `front`: USB RGB camera (640×480 @ 30 Hz, MJPG)
- `side`: USB RGB camera (640×480 @ 30 Hz, MJPG)

## Files

| File | Purpose |
| :--- | :--- |
| `rebot_config.py` | GR00T N1.7 modality config (`NEW_EMBODIMENT`, 3 cameras, 16-step action horizon) |
| `modality.json` | LeRobot v2 dataset `meta/modality.json` mapping template |
| `rebot_control_utils.py` | Hardware-independent command limits, chunk blending, EMA, velocity/accel limiter, and async episode recorder |
| `eval_rebot_arm_rs_3cam_v2.py` | Closed-loop real-robot evaluation client (preview + execute + async video/telemetry recording) |
| `eval_rebot_arm_rs_3cam.py` | Thin compatibility wrapper forwarding to `eval_rebot_arm_rs_3cam_v2.py` |

## 1. Dataset Preparation (LeRobot v3 → v2)

```bash
# Convert a backup copy of your LeRobot v3 dataset to v2 for GR00T
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id <path-to-lerobot-dataset-copy>

# Copy the 3-camera modality mapping into meta/
cp examples/rebot_arm/modality.json <path-to-lerobot-dataset-copy>/meta/modality.json
```

### Why `rebot_config.py` Uses `ABSOLUTE` for `single_arm`

In B601-RS LeRobot datasets, `observation.state` records physical motor angles while `action` records command angles with sign/scale mapping `JOINT_DIRECTIONS = [1, 1, -1, -1, -1, 1, 6]`. Because `elbow_flex`, `wrist_flex`, and `wrist_yaw` have opposite signs between `state` and `action`, setting `ActionRepresentation.RELATIVE` computes `action - state ≈ -2 * state`, doubling the normalization range on those three joints (~200° vs ~100°) and degrading final-approach precision. Using `ActionRepresentation.ABSOLUTE` doubles normalization resolution on those joints without modifying recorded datasets or the hardware driver.

## 2. Fine-Tuning GR00T N1.7

```bash
source .venv/bin/activate

export NUM_GPUS=1
export GLOBAL_BATCH_SIZE=32
export DATALOADER_NUM_WORKERS=4
export MAX_STEPS=15000
export SAVE_STEPS=5000
export USE_WANDB=1

bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <path-to-lerobot-v2-dataset> \
  --modality-config-path examples/rebot_arm/rebot_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir outputs/groot_rebot_b601_rs_3cam \
  --state-dropout-prob 0.1 \
  --use-percentiles false \
  --color-jitter-params "brightness 0.2 contrast 0.25 saturation 0.3 hue 0.04" \
  -- \
  --tune-top-llm-layers 2
```

Notes:
- `--use-percentiles false` prevents `q01`/`q99` from hard-clipping legitimate workspace edge poses (e.g., far-right `shoulder_pan` and `wrist_roll`). Use `true` only if your dataset contains corrupted outlier spikes.
- When `--tune-top-llm-layers 2` is enabled on a 48 GB GPU with `GLOBAL_BATCH_SIZE=32`, peak VRAM is ~44 GB; on smaller GPUs add `--gradient-checkpointing` after `--` or use `GLOBAL_BATCH_SIZE=16` with `--gradient-accumulation-steps 2`.
- After training completes, processor artifacts (`processor_config.json`, `statistics.json`, `embodiment_id.json`) are saved both inside each `checkpoint-*` and in the root `--output-dir`.

## 3. Closed-Loop Real-Robot Deployment

### Step 1: Bring up CAN bus & verify camera devices

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
v4l2-ctl --list-devices
```

### Step 2: Start the GR00T Policy Server (`Isaac-GR00T` `.venv`)

```bash
source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/groot_rebot_b601_rs_3cam \
  --embodiment-tag NEW_EMBODIMENT \
  --denoising-steps 4
```

The server automatically runs one CUDA warmup pass before listening (`--warmup` defaults to `True`) to eliminate first-request latency jitter.

### Step 3: Preview First (Mandatory Safety Check, `lerobot` env)

```bash
conda activate lerobot
python examples/rebot_arm/eval_rebot_arm_rs_3cam_v2.py \
  --front-camera /dev/video0 --side-camera /dev/video2
```

Preview mode waits until all 7 motors return live CAN feedback (`motor.get_state() is not None`), prints `Startup physical state (deg)`, disables torque, and prints one predicted `first action` without driving the arm.

### Step 4: Execute Closed-Loop Evaluation (`lerobot` env)

```bash
python examples/rebot_arm/eval_rebot_arm_rs_3cam_v2.py \
  --execute --duration-s 60 \
  --front-camera /dev/video0 --side-camera /dev/video2 \
  --record-dir outputs/eval_videos/run01
```

Each recorded run asynchronously saves 3-camera MP4 videos (`*_front.mp4`, `*_side.mp4`, `*_wrist.mp4`), commanded actions (`*_actions.npy`), sampled command-space states (`*_states.npy`), and timing/latency telemetry (`*_meta.json`) without blocking the 30 Hz control loop.

## 4. Control Tuning & Safety Notes

- **Validated critically damped defaults** (tuned against ~±0.8° chunk-to-chunk diffusion variance):
  - `--action-horizon 8`, `--latency-steps 4` (4-step overlap between consecutive chunks)
  - `--chunk-blend-weight 0.85`, `--ema-alpha 0.40`, `--tracking-gain 0.35`
  - `--gripper-delay-steps 0` (off by default so learned grasp timing is preserved)
- **Why `--use-rtc` is off by default (experimental)**:
  - Upstream GR00T N1.7's RTC implementation replaces the first `rtc_overlap_steps` tokens of initial flow-matching noise (`t = 0`) with clean previous actions and zeros velocity on `rtc_frozen_steps`. Because the ActionHead is not trained with clean-prefix inpainting, enabling `--use-rtc` in real-robot closed-loop control can induce self-locking at idle/home poses and rightward drift. Keep `--use-rtc` disabled for closed-loop B601-RS deployment.
- **Emergency stop**:
  - First `Ctrl+C`: smoothly interpolates back to startup pose, holds 2 s, then disables torque.
  - Second `Ctrl+C`: immediately requests torque disable.
- **Camera alignment**: Fixed grasp offsets usually indicate `front` or `side` camera bump/shift — align live views against reference frames from your training dataset before retuning.
