"""
Isaac Lab Environment Configuration — Spot Navigation
=======================================================
Defines the full environment config for Isaac Lab's DirectRLEnv.

This replaces mjx_nav_env.py. Physics runs via PhysX 5 (GPU).
Depth cameras use Isaac Sim RTX rendering.
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING

# Absolute path to Spot USD — works whether cwd is /workspace or /isaac-sim
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPOT_USD_PATH = os.path.join(_REPO_ROOT, "models", "spot_omniverse.usd")

from .config import (
    PHYSICS_DT, CONTROL_DT, PHYSICS_SUBSTEPS,
    JOINT_LOWER, JOINT_UPPER, STANDING_POSE, TARGET_HEIGHT,
    N_CAMS, CAM_H, CAM_W, H_FOV, V_FOV, MIN_DEPTH, MAX_DEPTH,
    N_STATIC, N_DYNAMIC, N_HUMANOID, N_OBS, OBS_HALF_SIZES,
    HUMANOID_OBSTACLE,
    ACTION_DIM, PROPRIO_DIM,
    PATCH_SIZE, TERRAIN_ROWS, TERRAIN_COLS,
)

# NOTE: These imports require Isaac Lab to be installed.
# Isaac Lab 2.0+ uses "isaaclab.*", older versions use "omni.isaac.lab.*".
# Try both to support whatever version is installed in the container.
HAS_ISAAC = False
_ISAAC_IMPORT_ERROR = None

try:
    # Isaac Lab 2.0+ (standalone package)
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    try:
        from isaaclab.sensors import TiledCameraCfg as _ActiveCameraClass
    except ImportError:
        _ActiveCameraClass = CameraCfg
    from isaaclab.sim import SimulationCfg, PhysxCfg
    from isaaclab.utils import configclass
    # Terrain imports — all re-exported from isaaclab.terrains top-level
    # (isaaclab/terrains/__init__.py does wildcard imports from height_field)
    # Verified against Isaac Lab main branch source.
    from isaaclab.terrains import (
        TerrainImporterCfg,
        TerrainGeneratorCfg,
        HfRandomUniformTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfInvertedPyramidStairsTerrainCfg,  # confirmed class name in hf_terrains_cfg.py
    )
    HAS_ISAAC = True
    HAS_TERRAIN = True
except ImportError:
    try:
        # Isaac Lab 1.x (omniverse extension)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from omni.isaac.lab.envs import DirectRLEnvCfg
        from omni.isaac.lab.scene import InteractiveSceneCfg
        from omni.isaac.lab.sensors import CameraCfg
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as _ActiveCameraClass
        except ImportError:
            _ActiveCameraClass = CameraCfg
        from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
        from omni.isaac.lab.utils import configclass
        HAS_ISAAC = True
        HAS_TERRAIN = False
    except ImportError as _e:
        _ISAAC_IMPORT_ERROR = str(_e)
        HAS_TERRAIN = False

if not HAS_ISAAC:
    # Provide stub for development without Isaac Lab
    def configclass(cls):
        return cls


# ════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIG
# ════════════════════════════════════════════════════════════════════════════

if HAS_ISAAC:

    @configclass
    class SpotSimCfg(SimulationCfg):
        """PhysX 5 simulation settings matching MuJoCo physics behavior."""
        dt = PHYSICS_DT                    # 0.005s = 200 Hz
        render_interval = PHYSICS_SUBSTEPS  # render every 4 physics steps
        gravity = (0.0, 0.0, -9.81)
        physx: PhysxCfg = PhysxCfg(
            # GPU solver is the default in Isaac Lab 0.54+
            solver_type=1,                 # TGS solver (better for articulations)
            max_position_iteration_count=8,
            max_velocity_iteration_count=1,
            # Contact parameters tuned to approximate MuJoCo behavior
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.025,
            # GPU buffer sizes for parallel envs
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 25,
            gpu_total_aggregate_pairs_capacity=2 ** 21,
            gpu_max_rigid_contact_count=2 ** 23,
            gpu_max_rigid_patch_count=2 ** 23,
            gpu_heap_capacity=2 ** 26,
            gpu_temp_buffer_capacity=2 ** 24,
            gpu_max_num_partitions=8,
        )


    # ════════════════════════════════════════════════════════════════════════
    # SCENE CONFIG
    # ════════════════════════════════════════════════════════════════════════

    @configclass
    class SpotSceneCfg(InteractiveSceneCfg):
        """Scene with Spot robot, ground, and depth cameras."""

        # ── Terrain (4×4 curriculum grid, 8m patches) ───────────────
        # Row 0 = flat (easy), Row 3 = rough + stairs (hard).
        # Robots are promoted/demoted between rows by the curriculum.
        # Falls back to flat plane if terrain imports are unavailable.
        if HAS_TERRAIN:
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    seed=0,
                    size=(PATCH_SIZE, PATCH_SIZE),
                    border_width=0.25,
                    num_rows=TERRAIN_ROWS,
                    num_cols=TERRAIN_COLS,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    slope_threshold=0.75,
                    curriculum=True,
                    sub_terrains={
                        # 40 % flat (rows 0-1 priority)
                        "flat": HfRandomUniformTerrainCfg(
                            proportion=0.4,
                            noise_range=(0.0, 0.01),
                            noise_step=0.01,
                        ),
                        # 20 % rough
                        "rough": HfRandomUniformTerrainCfg(
                            proportion=0.2,
                            noise_range=(0.02, 0.10),
                            noise_step=0.02,
                        ),
                        # 20 % stairs up
                        "stairs_up": HfPyramidStairsTerrainCfg(
                            proportion=0.2,
                            step_height_range=(0.05, 0.23),
                            step_width=0.30,
                            platform_width=3.0,
                        ),
                        # 20 % stairs down
                        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
                            proportion=0.2,
                            step_height_range=(0.05, 0.23),
                            step_width=0.30,
                            platform_width=3.0,
                        ),
                    },
                ),
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
        else:
            # Flat-plane fallback (no terrain package)
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    size=(200.0, 200.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )

        # Spot robot (imported from MJCF -> USD)
        # The MJCF importer converts spot_scene.xml + OBJ meshes to USD.
        # After conversion, reference the USD path here.
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                # Path to converted USD (user must run MJCF import first)
                usd_path=SPOT_USD_PATH,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, TARGET_HEIGHT),
                joint_pos={
                    # Standing pose for all 12 joints
                    "fl_hx": STANDING_POSE[0],
                    "fl_hy": STANDING_POSE[1],
                    "fl_kn": STANDING_POSE[2],
                    "fr_hx": STANDING_POSE[3],
                    "fr_hy": STANDING_POSE[4],
                    "fr_kn": STANDING_POSE[5],
                    "hl_hx": STANDING_POSE[6],
                    "hl_hy": STANDING_POSE[7],
                    "hl_kn": STANDING_POSE[8],
                    "hr_hx": STANDING_POSE[9],
                    "hr_hy": STANDING_POSE[10],
                    "hr_kn": STANDING_POSE[11],
                },
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=500.0,      # kp matches MuJoCo
                    damping=40.0,         # kv matches MuJoCo
                    effort_limit_sim=1000.0,
                ),
            },
        )

        # ── Depth cameras (3 front-facing cameras on Spot body) ─────
        # Covers 180° forward arc. Rear cameras removed to reduce
        # RTX view count (128 envs × 3 = 384 vs 640 with 5 cameras).
        # Each camera: 120x160 pixels, 87 deg HFOV, depth only.
        cam_front_center = _ActiveCameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_center",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_front_left = _ActiveCameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_left",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )
        cam_front_right = _ActiveCameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/body/cam_front_right",
            update_period=CONTROL_DT,
            height=CAM_H,
            width=CAM_W,
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0,
                horizontal_aperture=2.0 * math.tan(math.radians(H_FOV / 2)),
                clipping_range=(MIN_DEPTH, MAX_DEPTH),
            ),
        )

        # ── Obstacle rigid bodies (2 static boxes + 1 humanoid slot) ──
        # Kinematic bodies — positions written each reset from SpotNavEnv._obs_pos.
        # Only placed on flat terrain patches (row ≤ FLAT_TERRAIN_ROW_MAX).
        # Spawned far off-scene initially; moved on reset for flat-terrain envs.
    # Build obstacle configs programmatically from OBS_HALF_SIZES
    def _build_obstacle_cfgs():
        """Generate RigidObjectCfg for each obstacle."""
        cfgs = {}
        for i in range(N_STATIC):
            hs = OBS_HALF_SIZES[i]
            cfgs[f"obs_static_{i:02d}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/obs_static_{i:02d}",
                spawn=sim_utils.CuboidCfg(
                    size=(hs[0] * 2, hs[1] * 2, hs[2] * 2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.4, 0.2),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(100.0, 0.0, hs[2]),
                ),
            )
        for i in range(N_DYNAMIC):
            idx = N_STATIC + i
            hs = OBS_HALF_SIZES[idx]
            cfgs[f"obs_dynamic_{i:02d}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/obs_dynamic_{i:02d}",
                spawn=sim_utils.CylinderCfg(
                    radius=hs[0],
                    height=hs[2] * 2,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.8),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(100.0, 0.0, hs[2]),
                ),
            )
        # Humanoid (approximated as a tall capsule/box)
        hs = OBS_HALF_SIZES[N_STATIC + N_DYNAMIC]
        cfgs["obs_humanoid"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/obs_humanoid",
            spawn=sim_utils.CapsuleCfg(
                radius=hs[0],
                height=hs[2] * 2 - hs[0] * 2,  # capsule height excludes end caps
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.3, 0.3),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(100.0, 0.0, HUMANOID_OBSTACLE["mocap_z"]),
            ),
        )
        return cfgs
    # Attach obstacle configs to scene class
    _obs_cfgs = _build_obstacle_cfgs()
    for _name, _cfg in _obs_cfgs.items():
        setattr(SpotSceneCfg, _name, _cfg)
    # Note: no ContactSensorCfg — collision detection uses distance-based
    # math in SpotNavEnv._get_rewards() (min_obs_dist < 0.35), not sensor data.
    # ════════════════════════════════════════════════════════════════════════
    # ENVIRONMENT CONFIG
    # ════════════════════════════════════════════════════════════════════════

    @configclass
    class SpotNavEnvCfg(DirectRLEnvCfg):
        """Full environment config for Spot navigation RL."""

        # Simulation
        sim: SimulationCfg = SpotSimCfg()
        decimation = PHYSICS_SUBSTEPS       # 4 physics steps per RL step

        # Scene
        scene: InteractiveSceneCfg = SpotSceneCfg(
            num_envs=4096,
            env_spacing=PATCH_SIZE,         # 8m matches terrain patch size
        )

        # Spaces (renamed from num_observations/num_actions in Isaac Lab 0.54)
        observation_space = PROPRIO_DIM      # 37 (depth handled separately via cameras)
        action_space = ACTION_DIM            # 12

        # Episode
        episode_length_s = 1000 * CONTROL_DT  # 1000 steps * 0.02s = 20s

else:
    # Stubs when Isaac Lab is not installed — raise ImportError on use
    # so train.py's except ImportError catches it and shows the real cause
    def _raise():
        raise ImportError(
            f"omni.isaac.lab is not available.\n"
            f"  Root cause: {_ISAAC_IMPORT_ERROR}\n"
            f"  Verify Isaac Lab is installed: "
            f"/isaac-sim/python.sh -c \"import omni.isaac.lab; print('OK')\""
        )

    class SpotSimCfg:
        def __init__(self, *a, **kw): _raise()

    class SpotSceneCfg:
        def __init__(self, *a, **kw): _raise()

    class SpotNavEnvCfg:
        def __init__(self, *a, **kw): _raise()
