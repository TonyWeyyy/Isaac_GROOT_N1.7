from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


rebot_3cam_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist", "front", "side"],  # wrist egocentric + front/side third-person
    ),
    # State: current proprioceptive reading; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # 6 arm joint positions
            "gripper",     # gripper state
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: ABSOLUTE target joint commands.
            # NOTE: In Seeed reBot B601-RS datasets, observation.state (physical motor angles)
            # and action (command angles) have opposite signs on elbow_flex/wrist_flex/wrist_yaw
            # (JOINT_DIRECTIONS = [1, 1, -1, -1, -1, 1, 6]). Using RELATIVE computes
            # action - state ≈ -2 * state on those 3 joints, doubling the normalization span
            # (~200 deg vs ~100 deg) and hurting 1-2 cm final-approach precision.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target position (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(rebot_3cam_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)