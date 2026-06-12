"""
Isaac Lab Environment Config Builder — robot-agnostic
======================================================
Builds SimulationCfg / InteractiveSceneCfg / DirectRLEnvCfg at runtime from
an ExperimentCfg (dataclasses in configs/). Nothing robot-specific lives
here; scene entities (robot, terrain, obstacles, sensors, cameras) are
attached as INSTANCE attributes of the scene cfg — InteractiveScene iterates
the instance __dict__, so dynamically attached entities are picked up.

Cameras are attached ONLY when exp_cfg.camera.enabled — Phase 1 constructs
zero camera prims and never needs --enable_cameras.

Keeps the repo's dual-import pattern: Isaac Lab 2.x (isaaclab.*) with
fallback to 1.x (omni.isaac.lab.*), and stub classes that raise a clear
ImportError when neither is installed.
"""

from __future__ import annotations

import math

from .configs.base import ExperimentCfg

# NOTE: These imports require Isaac Lab AND a running simulation app
# (train.py launches AppLauncher before importing this module).
HAS_ISAAC = False
HAS_TERRAIN = False
_ISAAC_IMPORT_ERROR: str | None = None

try:
    # Isaac Lab 2.x (standalone package)
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg
    try:
        from isaaclab.sensors import TiledCameraCfg as _ActiveCameraClass
    except ImportError:
        _ActiveCameraClass = CameraCfg
    try:
        from isaaclab.sensors.patterns import GridPatternCfg
    except ImportError:
        from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass
    try:
        from isaaclab.terrains import (
            HfInvertedPyramidStairsTerrainCfg,
            HfPyramidStairsTerrainCfg,
            HfRandomUniformTerrainCfg,
            TerrainGeneratorCfg,
            TerrainImporterCfg,
        )
        HAS_TERRAIN = True
    except ImportError:
        HAS_TERRAIN = False
    HAS_ISAAC = True
except ImportError:
    try:
        # Isaac Lab 1.x (omniverse extension)
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import (
            ArticulationCfg, AssetBaseCfg, RigidObjectCfg,
        )
        from omni.isaac.lab.envs import DirectRLEnvCfg
        from omni.isaac.lab.scene import InteractiveSceneCfg
        from omni.isaac.lab.sensors import (
            CameraCfg, ContactSensorCfg, RayCasterCfg,
        )
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as _ActiveCameraClass
        except ImportError:
            _ActiveCameraClass = CameraCfg
        try:
            from omni.isaac.lab.sensors.patterns import GridPatternCfg
        except ImportError:
            from omni.isaac.lab.sensors.ray_caster.patterns import GridPatternCfg
        from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
        from omni.isaac.lab.utils import configclass
        try:
            from omni.isaac.lab.terrains import (
                HfInvertedPyramidStairsTerrainCfg,
                HfPyramidStairsTerrainCfg,
                HfRandomUniformTerrainCfg,
                TerrainGeneratorCfg,
                TerrainImporterCfg,
            )
            HAS_TERRAIN = True
        except ImportError:
            HAS_TERRAIN = False
        HAS_ISAAC = True
    except ImportError as _e:
        _ISAAC_IMPORT_ERROR = str(_e)


if HAS_ISAAC:

    # ── Base env cfg class (fields overwritten by build_env_cfg) ─────────
    @configclass
    class NavEnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 20.0
        sim: SimulationCfg = SimulationCfg()
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4, env_spacing=8.0
        )
        observation_space = 1   # informational (obs is a dict; see NavEnv)
        action_space = 1
        state_space = 0

    # ── Builders ─────────────────────────────────────────────────────────

    def _build_sim_cfg(x: ExperimentCfg) -> SimulationCfg:
        """PhysX settings preserved from the tuned SpotSimCfg."""
        return SimulationCfg(
            dt=x.sim.physics_dt,
            render_interval=x.sim.decimation,
            gravity=(0.0, 0.0, -9.81),
            physx=PhysxCfg(
                solver_type=1,  # TGS
                max_position_iteration_count=8,
                max_velocity_iteration_count=1,
                bounce_threshold_velocity=0.5,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.025,
                gpu_found_lost_pairs_capacity=2**21,
                gpu_found_lost_aggregate_pairs_capacity=2**25,
                gpu_total_aggregate_pairs_capacity=2**21,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
                gpu_heap_capacity=2**26,
                gpu_temp_buffer_capacity=2**24,
                gpu_collision_stack_size=2**28,
                gpu_max_num_partitions=8,
            ),
        )

    def _build_terrain(x: ExperimentCfg):
        t = x.terrain
        if not HAS_TERRAIN:
            return AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    size=(200.0, 200.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=t.seed,
                size=(t.patch_size, t.patch_size),
                border_width=t.border_width,
                num_rows=t.rows,
                num_cols=t.cols,
                horizontal_scale=t.horizontal_scale,
                vertical_scale=t.vertical_scale,
                slope_threshold=t.slope_threshold,
                curriculum=True,
                sub_terrains={
                    "flat": HfRandomUniformTerrainCfg(
                        proportion=t.flat_proportion,
                        noise_range=t.flat_noise_range,
                        noise_step=t.flat_noise_step,
                    ),
                    "rough": HfRandomUniformTerrainCfg(
                        proportion=t.rough_proportion,
                        noise_range=t.rough_noise_range,
                        noise_step=t.rough_noise_step,
                    ),
                    "stairs_up": HfPyramidStairsTerrainCfg(
                        proportion=t.stairs_up_proportion,
                        step_height_range=t.stair_step_height_range,
                        step_width=t.stair_step_width,
                        platform_width=t.stair_platform_width,
                    ),
                    "stairs_down": HfInvertedPyramidStairsTerrainCfg(
                        proportion=t.stairs_down_proportion,
                        step_height_range=t.stair_step_height_range,
                        step_width=t.stair_step_width,
                        platform_width=t.stair_platform_width,
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

    def _build_robot(x: ExperimentCfg) -> ArticulationCfg:
        r = x.robot
        actuator_kwargs = dict(
            joint_names_expr=[".*"],
            stiffness=r.actuator_stiffness,
            damping=r.actuator_damping,
        )
        try:
            actuators = {"legs": ImplicitActuatorCfg(
                **actuator_kwargs, effort_limit_sim=r.effort_limit,
            )}
        except TypeError:  # older field name
            actuators = {"legs": ImplicitActuatorCfg(
                **actuator_kwargs, effort_limit=r.effort_limit,
            )}
        return ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=r.usd_path,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, r.init_height),
                joint_pos=dict(zip(r.joint_names, r.default_joint_pos)),
            ),
            actuators=actuators,
        )

    def _build_obstacles(x: ExperimentCfg) -> dict:
        cfgs = {}
        for i in range(x.obstacles.n_static):
            hs = x.obstacles.half_sizes[i]
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
        return cfgs

    def _build_height_scanner(x: ExperimentCfg) -> RayCasterCfg:
        sc = x.scandots
        kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/" + x.robot.base_body_name,
            offset=RayCasterCfg.OffsetCfg(
                pos=(sc.forward_offset, 0.0, 20.0)
            ),
            pattern_cfg=GridPatternCfg(
                resolution=sc.spacing, size=sc.size
            ),
            mesh_prim_paths=["/World/ground"],
            update_period=0.0,
            debug_vis=False,
        )
        try:
            return RayCasterCfg(
                **kwargs,
                ray_alignment="yaw" if sc.attach_yaw_only else "base",
            )
        except TypeError:  # older versions only know attach_yaw_only
            return RayCasterCfg(**kwargs, attach_yaw_only=sc.attach_yaw_only)

    def _build_contact_sensor(x: ExperimentCfg) -> ContactSensorCfg:
        return ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/" + x.robot.contact_body_regex
            ),
            update_period=0.0,
        )

    def _build_cameras(x: ExperimentCfg) -> dict:
        cam = x.camera
        cfgs = {}
        cls = _ActiveCameraClass
        for mount in cam.mounts:
            cfgs[mount.name] = cls(
                prim_path=(
                    f"{{ENV_REGEX_NS}}/Robot/{x.robot.cam_mount_body}"
                    f"/{mount.name}"
                ),
                update_period=cam.update_period_s,
                height=cam.height,
                width=cam.width,
                data_types=[cam.data_type],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.0,
                    horizontal_aperture=(
                        2.0 * math.tan(math.radians(cam.h_fov_deg / 2))
                    ),
                    clipping_range=(cam.min_depth, cam.max_depth),
                ),
                offset=cls.OffsetCfg(
                    pos=mount.pos, rot=mount.rot, convention=mount.convention,
                ),
            )
        return cfgs

    def build_env_cfg(x: ExperimentCfg, num_envs: int) -> NavEnvCfg:
        """Assemble the full DirectRLEnvCfg from the experiment config."""
        scene = InteractiveSceneCfg(
            num_envs=num_envs, env_spacing=x.terrain.patch_size
        )
        # Insertion order matters: robot before sensors that attach to it.
        if HAS_TERRAIN:
            scene.terrain = _build_terrain(x)
        else:
            scene.ground = _build_terrain(x)
        scene.robot = _build_robot(x)
        for name, cfg in _build_obstacles(x).items():
            setattr(scene, name, cfg)
        scene.height_scanner = _build_height_scanner(x)
        scene.contact_sensor = _build_contact_sensor(x)
        if x.camera.enabled:
            for name, cfg in _build_cameras(x).items():
                setattr(scene, name, cfg)

        env_cfg = NavEnvCfg()
        env_cfg.sim = _build_sim_cfg(x)
        env_cfg.scene = scene
        env_cfg.decimation = x.sim.decimation
        env_cfg.episode_length_s = (
            x.goal.episode_len_steps * x.sim.control_dt
        )
        env_cfg.observation_space = x.proprio_dim
        env_cfg.action_space = x.action_dim
        # Older Isaac Lab versions use num_observations/num_actions
        for old, new in (("num_observations", x.proprio_dim),
                         ("num_actions", x.action_dim),
                         ("num_states", 0)):
            if hasattr(env_cfg, old):
                setattr(env_cfg, old, new)
        return env_cfg

else:
    # Stubs so imports fail with the real cause at use time
    def _raise():
        raise ImportError(
            "Isaac Lab is not available.\n"
            f"  Root cause: {_ISAAC_IMPORT_ERROR}\n"
            "  Install Isaac Lab: https://isaac-sim.github.io/IsaacLab/"
        )

    class NavEnvCfg:  # noqa: D401
        def __init__(self, *a, **kw):
            _raise()

    def build_env_cfg(x: ExperimentCfg, num_envs: int):
        _raise()
