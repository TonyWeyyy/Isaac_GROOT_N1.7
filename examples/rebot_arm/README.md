# Finetuning & Deploying GR00T N1.7 on Seeed reBot Arm B601-RS (3 Cameras)

This directory contains the modality configuration, dataset command-space alignment script, hardware-independent safety/control utilities, and closed-loop real-robot evaluation client for the **Seeed reBot Arm B601-RS** (RobStride motors over SocketCAN) with a 3-camera setup:

- `wrist`: Orbbec Gemini 2 RGB (bound by serial number)
- `front`: USB RGB camera (640×480 @ 30 Hz, MJPG)
- `side`: USB RGB camera (640×480 @ 30 Hz, MJPG)

## Files

| File | Purpose |
| :--- | :--- |
| `align_dataset_command_space.py` | Non-destructive converter aligning `observation.state` to command space (`state / JOINT_DIRECTIONS`) for true `RELATIVE` actions |
| `rebot_config.py` | GR00T N1.7 modality config (`NEW_EMBODIMENT`, 3 cameras, true command-space `RELATIVE` arm + `ABSOLUTE` gripper, 16-step horizon) |
| `modality.json` | LeRobot v2 dataset `meta/modality.json` mapping template |
| `rebot_control_utils.py` | Hardware-independent command limits, chunk blending, EMA, velocity/accel limiter, startup camera alignment check, and async episode recorder |
| `eval_rebot_arm_rs_3cam_v2.py` | Closed-loop real-robot evaluation client (preview + camera alignment + execute + `--trt-fast` + async video/telemetry recording) |
| `eval_rebot_arm_rs_3cam.py` | Thin compatibility wrapper forwarding to `eval_rebot_arm_rs_3cam_v2.py` |

## 1. Dataset Preparation (LeRobot v3 → v2 + Command-Space Alignment)

In raw B601-RS LeRobot recordings, `observation.state` stores physical motor angles while `action` stores command angles (`command = physical / JOINT_DIRECTIONS`, where `JOINT_DIRECTIONS = [1, 1, -1, -1, -1, 1, 6]`). Aligning `observation.state` to command space shrinks the step-0 `action - state` normalization span on `elbow_flex` and `wrist_flex` from **~222° / ~206°** down to **~22.9° / ~21.8°** (~10× higher resolution for final 1–2 cm grasping):

```bash
# 1. Convert a backup copy of your LeRobot v3 dataset to v2
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id <path-to-lerobot-v2-dataset>

# 2. Place the 3-camera modality mapping in meta/
cp examples/rebot_arm/modality.json <path-to-lerobot-v2-dataset>/meta/modality.json

# 3. Create a command-space-aligned copy (symlinks videos/, zero extra video disk usage)
source .venv/bin/activate
python examples/rebot_arm/align_dataset_command_space.py \
  --src <path-to-lerobot-v2-dataset> \
  --dst <path-to-lerobot-v2-cmd-rel-dataset>
```

## 2. Fine-Tuning GR00T N1.7 (320×426 High-Res + True Relative + Full Workspace Reach)

```bash
source .venv/bin/activate

export NUM_GPUS=1
export GLOBAL_BATCH_SIZE=16
export DATALOADER_NUM_WORKERS=4
export MAX_STEPS=15000
export SAVE_STEPS=5000
export USE_WANDB=1

bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <path-to-lerobot-v2-cmd-rel-dataset> \
  --modality-config-path examples/rebot_arm/rebot_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir outputs/groot_grab_red_cube_3cam_v3_cmd_rel_320 \
  --state-dropout-prob 0.1 \
  --use-percentiles false \
  --shortest-image-edge 320 \
  --crop-fraction 0.95 \
  --color-jitter-params "brightness 0.2 contrast 0.25 saturation 0.3 hue 0.04" \
  -- \
  --tune-top-llm-layers 2 \
  --gradient-accumulation-steps 2
```

Key design choices:
- `--shortest-image-edge 320 --crop-fraction 0.95`: produces `320×426` inputs per camera (+56.6% pixel area vs default `256×340`) for sharper wrist-camera edges.
- `GLOBAL_BATCH_SIZE=16` with `--gradient-accumulation-steps 2`: keeps effective batch size `32` while staying around `44.2 / 49.1 GB` VRAM on RTX 4090 D 48 GB.
- `--use-percentiles false`: prevents `q01`/`q99` from clipping legitimate workspace edge poses.

## 3. TensorRT Acceleration (RTX 4090 / dGPU)

Export ONNX, build full-pipeline optimized TensorRT engines (ViT FP32 + LLM BF16 + Action Head CUDA Graph), verify PyTorch-vs-TRT parity (`cosine > 0.997`), and benchmark end-to-end latency:

```bash
source .venv/bin/activate
export PATH="/opt/tensorrt/bin:$PATH"

python scripts/deployment/build_trt_pipeline.py \
  --model-path outputs/groot_grab_red_cube_3cam_v3_cmd_rel_320 \
  --dataset-path <path-to-lerobot-v2-cmd-rel-dataset> \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir outputs/trt_groot_grab_red_cube_3cam_v3_cmd_rel_320 \
  --export-mode full_pipeline \
  --execution-profile optimized \
  --quantization none \
  --steps all \
  --skip-compile \
  --flat-e2e
```

Verified RTX 4090 D latency (3× `320×426` cameras, 4 denoising steps, full E2E including preprocessing & decoding):
- **PyTorch Eager**: `134.9 ms` median (max `192.4 ms`)
- **TensorRT Optimized (CUDA Graph + thread-contention fix)**: **`52.2 ms` p50, `54.3 ms` p95, `59.0 ms` max (19.2 Hz)**

## 4. Closed-Loop Real-Robot Deployment

### Step 1: Bring up CAN bus & verify camera indices

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
v4l2-ctl --list-devices
```

### Step 2: Start the TensorRT Policy Server (`Isaac-GR00T` `.venv`)

```bash
source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/groot_grab_red_cube_3cam_v3_cmd_rel_320 \
  --trt-engine-path outputs/trt_groot_grab_red_cube_3cam_v3_cmd_rel_320/engines \
  --trt-mode n17_full_pipeline \
  --embodiment-tag NEW_EMBODIMENT \
  --denoising-steps 4 \
  --warmup-instruction "Pick up the red cube and put it into the box"
```

### Step 3: Preview & Camera Alignment Check (Mandatory Before `--execute`, `lerobot` env)

```bash
conda activate lerobot
python examples/rebot_arm/eval_rebot_arm_rs_3cam_v2.py \
  --front-camera /dev/video0 --side-camera /dev/video2 \
  --reference-dataset <path-to-lerobot-v2-cmd-rel-dataset>
```

Preview mode verifies live CAN feedback from all 7 motors, compares live `front`/`side`/`wrist` frames against training episode 0 (saving a 3-row overlay to `outputs/camera_alignment_check.jpg` and warning if a static camera shifted), disables torque, and prints one predicted `first action`.

### Step 4: Execute Closed-Loop Evaluation (`lerobot` env)

```bash
python examples/rebot_arm/eval_rebot_arm_rs_3cam_v2.py \
  --execute --trt-fast --duration-s 60 \
  --front-camera /dev/video0 --side-camera /dev/video2 \
  --reference-dataset <path-to-lerobot-v2-cmd-rel-dataset> \
  --record-dir outputs/eval_videos/cmd_rel_320_trt_test
```

## 5. Control, Compatibility & Safety Notes

- **`--state-space`**: defaults to `command` (required for models trained on `align_dataset_command_space.py`). Pass `--state-space physical` only when evaluating older models trained on unaligned physical-state datasets.
- **Validated critically damped defaults**: `ema_alpha=0.40`, `chunk_blend_weight=0.85` (`6`-step decay), `tracking_gain=0.35`, `gripper_delay_steps=0`, with transit speed cap `TRAJECTORY_MAX_STEP = [1.4, 1.2, 1.2, 1.2, 1.4, 1.4]` deg/tick (≤ 0.73 rad/s) and smooth accel limits `TRAJECTORY_MAX_STEP_CHANGE = [0.12, 0.10, 0.10, 0.10, 0.12, 0.12]`.
- **`--trt-fast` preset**: sets `--action-horizon 6 --latency-steps 3` (100 ms async budget > 59 ms max TRT latency, 5 Hz replanning, full 6-step overlap).
- **`--use-rtc` (experimental, disabled by default)**: upstream N1.7 naive clean-prefix inpainting can cause closed-loop self-locking; keep disabled for real-robot grasping.
- **Emergency stop**: first `Ctrl+C` smoothly returns to startup pose and disables torque; second `Ctrl+C` requests immediate torque disable.
