"""
Spot Robot Configuration
=========================
All Boston Dynamics Spot specific values (migrated from the old config.py).
This module is the template for adding new robots: copy it, change values,
and select it with `--robot <module_name>`.
"""

from __future__ import annotations

import os

from .base import (
    CameraMountCfg,
    CameraRigCfg,
    ExperimentCfg,
    RobotCfg,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def make_cfg() -> ExperimentCfg:
    cfg = ExperimentCfg()

    cfg.robot = RobotCfg(
        name="spot",
        usd_path=os.path.join(_REPO_ROOT, "models", "spot_omniverse.usd"),
        joint_names=(
            "fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",
        ),
        # Real Spot URDF limits (rad), in joint_names order
        joint_lower=(
            -0.785398, -0.89012, -2.7929,
            -0.785398, -0.89012, -2.7929,
            -0.785398, -0.89012, -2.7929,
            -0.785398, -0.89012, -2.7929,
        ),
        joint_upper=(
            0.785398, 2.29511, -0.254402,
            0.785398, 2.29511, -0.255648,
            0.785398, 2.29511, -0.247067,
            0.785398, 2.29511, -0.248282,
        ),
        joint_vel_limits=(
            4.0, 4.0, 6.0,
            4.0, 4.0, 6.0,
            4.0, 4.0, 6.0,
            4.0, 4.0, 6.0,
        ),
        default_joint_pos=(0.0, 1.04, -1.8) * 4,  # standing pose
        action_scale=None,  # half joint range (original behavior)
        base_body_name="body",
        contact_body_regex=".*_lleg",  # fl_lleg, fr_lleg, hl_lleg, hr_lleg
        num_feet=4,
        cam_mount_body="body",
        actuator_stiffness=500.0,
        actuator_damping=40.0,
        effort_limit=1000.0,
        init_height=0.5,
        mass_kg=32.7,
    )

    # Single forward depth camera (Extreme Parkour style). The previous
    # 3-camera 120x160 rig is expressible by listing 3 mounts and changing
    # width/height — no code changes.
    cfg.camera = CameraRigCfg(
        enabled=False,  # flipped on by --phase student
        width=87,
        height=58,
        update_period_s=0.1,
        h_fov_deg=87.0,
        min_depth=0.1,
        max_depth=10.0,
        mounts=(
            CameraMountCfg(
                name="cam_front_center",
                pos=(0.45, 0.0, 0.05),
                rot=(0.5, -0.5, 0.5, -0.5),  # optical axis = body +X (ros)
                convention="ros",
            ),
        ),
    )

    return cfg
