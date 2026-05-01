"""
PhysX 5 ↔ MuJoCo Physics Tuning Guide & Utilities
====================================================
This module documents the sim-to-sim gap between MuJoCo (generalized-coordinate,
elliptic cone friction, implicitfast integrator) and PhysX 5 (maximal-coordinate,
Coulomb pyramid friction, TGS solver) — and provides tunable overrides.

The goal is to make Spot walk similarly in both engines so trained policies
transfer. Perfect parity is impossible, but we can minimize the gap.

Usage:
    from omni_spot.physics_tuning import PHYSX_TUNING, apply_tuning

Key Differences
===============

1. CONTACT MODEL
   MuJoCo: elliptic friction cone with softness (solref, solimp).
           Contacts are "soft" — penetration absorbs energy gradually.
   PhysX:  Coulomb pyramid (4+ edge approximation). Contacts are "hard"
           — bodies bounce off with restitution. No soft contact layer.

   Impact: Spot's feet feel "stickier" in MuJoCo due to soft contacts.
           In PhysX, feet may slide more on ground → need higher friction.

2. FRICTION MODEL
   MuJoCo: condim=6 on feet = full torsional + rolling friction.
           friction="0.8 0.02 0.01" → tangent=0.8, torsional=0.02, rolling=0.01
   PhysX:  Only tangential friction (no torsional/rolling by default).
           Combine mode: default is "average" of two materials.

   Impact: Feet slip more in PhysX → increase static/dynamic friction.

3. SOLVER
   MuJoCo: implicitfast (Newton) — unconditionally stable, 1 solve.
   PhysX:  TGS (temporal Gauss-Seidel) — iterative, needs 4-8 position
           iterations for articulations to converge.

   Impact: Joints may feel "loose" with too few iterations. 8 pos iters
           is good for quadrupeds. More iters = stiffer but slower.

4. JOINT DRIVES
   MuJoCo: position actuators with kp=500, kv=40, force limit ±1000N.
           These are implicit PD controllers applied in the integrator.
   PhysX:  ImplicitActuator in Isaac Lab wraps PhysX articulation drives.
           stiffness=kp, damping=kv. Effort limit applies as torque cap.

   Impact: Nearly identical behavior. The main difference is that PhysX
           drives are applied per-substep while MuJoCo applies per-step.
           With 4 substeps at dt=0.005, this is the same effective rate.

5. MASS / INERTIA
   MuJoCo: generalized coordinates — inertia matrix is computed from
           the articulation tree. diaginertia values are principal moments.
   PhysX:  maximal coordinates — each body has its own 6-DOF state.
           Mass properties are set per-link in the USD.

   Impact: MJCF importer preserves mass/inertia from XML. Should match
           exactly if import_inertia_tensor=True.

6. GROUND CONTACT
   MuJoCo: Soft floor plane with condim=3, default solref/solimp.
   PhysX:  Rigid ground plane. Contact is hard unless compliance is added.

   Impact: Robot may "bounce" more on PhysX ground. Reduce restitution
           to zero and increase ground friction.
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# TUNING PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

PHYSX_TUNING = {
    # ── Ground material ─────────────────────────────────────────────
    # MuJoCo floor: condim=3, default friction ~1.0
    # PhysX: no torsional friction → compensate with higher tangential
    "ground_static_friction": 1.2,      # MuJoCo effective ~0.8, boost for PhysX
    "ground_dynamic_friction": 1.0,     # Slightly less than static
    "ground_restitution": 0.0,          # Zero bounce (MuJoCo has soft contacts)

    # ── Foot material ───────────────────────────────────────────────
    # MuJoCo foot: friction="0.8 0.02 0.01", condim=6, soft contact
    # PhysX: only tangential friction, no soft contact layer
    "foot_static_friction": 1.5,        # Higher than MuJoCo 0.8 to compensate
    "foot_dynamic_friction": 1.2,       # for lack of torsional/rolling friction
    "foot_restitution": 0.0,            # Soft landing (MuJoCo solimp absorbs energy)

    # ── Solver ──────────────────────────────────────────────────────
    # MuJoCo: 1 Newton solve (implicitfast)
    # PhysX TGS: iterative, needs more iters for stiff contacts
    "solver_position_iters": 8,         # 4=fast/loose, 8=good, 16=very stiff
    "solver_velocity_iters": 1,         # 1 is fine for position-controlled robot

    # ── Contact tuning ──────────────────────────────────────────────
    # These PhysX-specific parameters approximate MuJoCo's soft contacts
    "bounce_threshold_velocity": 0.5,   # Below this, no bounce (m/s)
    "friction_offset_threshold": 0.01,  # Contact offset for smooth friction
    "friction_correlation_distance": 0.025,  # Friction averaging distance

    # ── Joint drive scaling ─────────────────────────────────────────
    # MuJoCo kp=500, kv=40. May need slight adjustment in PhysX because:
    #  - PhysX applies drives per-substep (same dt=0.005 → should match)
    #  - But maximal-coordinate dynamics may respond differently
    "joint_stiffness": 500.0,           # Start same as MuJoCo kp
    "joint_damping": 40.0,              # Start same as MuJoCo kv
    "effort_limit": 1000.0,             # Same as MuJoCo actuatorfrcrange

    # ── Self-collision ──────────────────────────────────────────────
    # MuJoCo: explicit <contact><exclude> for body↔upper legs
    # PhysX: disable collision between adjacent links via USD collision filters
    "disable_adjacent_collisions": True,

    # ── Reward weight adjustments ───────────────────────────────────
    # These are MULTIPLIERS applied on top of the base reward weights
    # from config.py. Start at 1.0 (no change), tune after first runs.
    "reward_scale_progress": 1.0,
    "reward_scale_energy": 1.0,         # May need 1.2-1.5 if PhysX joints are stiffer
    "reward_scale_smooth": 1.0,
    "reward_scale_upright": 1.0,
    "reward_scale_height": 1.0,
    "reward_scale_collision": 1.0,
}

# ════════════════════════════════════════════════════════════════════════════
# TUNING PROCEDURE (step-by-step guide for the user)
# ════════════════════════════════════════════════════════════════════════════

TUNING_PROCEDURE = """
PhysX vs MuJoCo Tuning Procedure
==================================

STEP 1: STATIC STABILITY TEST (no RL, just physics)
  - Spawn Spot at standing pose in PhysX (home keyframe)
  - Run 1000 physics steps with zero control (all joints at standing pose)
  - Observe: Does Spot stay upright? How much does it drift?
  - In MuJoCo: Spot stays rock-solid at standing pose with zero drift.
  - If PhysX drifts: increase solver_position_iters (try 12-16)
  - If PhysX bounces on ground: check ground_restitution=0

STEP 2: WALKING GAIT TEST (replay MuJoCo trajectories)
  - Record 100 steps of joint positions from a trained MuJoCo policy
  - Replay those exact joint targets in PhysX (open-loop)
  - Compare: body height, forward velocity, foot slip
  - If feet slip more: increase foot_static_friction (try 2.0)
  - If body height differs: check mass/inertia imported correctly
  - If gait looks "mushy": increase joint_stiffness (try 600-800)

STEP 3: CONTACT FORCE COMPARISON
  - In MuJoCo: record foot contact forces during walking
  - In PhysX: record foot contact forces for same joint trajectory
  - Compare peak forces, contact duration, ground reaction pattern
  - Large differences indicate friction/restitution mismatch

STEP 4: REWARD SENSITIVITY ANALYSIS
  - Train for 500 iterations in PhysX with base reward weights
  - Check diagnostics: which reward terms dominate?
  - If r_energy is too large: robot is fighting stiff PhysX joints
    → reduce energy weight or reduce joint_stiffness
  - If r_smooth is too large: PhysX solver is amplifying jitter
    → increase solver_position_iters
  - If r_progress is too low: feet are slipping, robot can't move forward
    → increase foot friction

STEP 5: FINE-TUNE REWARD SCALES
  - Use PHYSX_TUNING["reward_scale_*"] multipliers
  - Goal: reward component magnitudes match MuJoCo training profile
  - Example: if MuJoCo r_energy averaged -0.3 but PhysX gives -0.5,
    set reward_scale_energy = 0.6 to compensate
"""


def apply_tuning(sim_cfg, scene_cfg):
    """Apply PHYSX_TUNING values to Isaac Lab sim/scene configs.

    Call this in train.py before creating the environment:
        from omni_spot.physics_tuning import apply_tuning
        apply_tuning(env_cfg.sim, env_cfg.scene)

    Parameters
    ----------
    sim_cfg : SpotSimCfg
        The simulation config (PhysX solver settings).
    scene_cfg : SpotSceneCfg
        The scene config (robot actuator settings).
    """
    t = PHYSX_TUNING

    # Solver iterations
    sim_cfg.physx.max_position_iteration_count = t["solver_position_iters"]
    sim_cfg.physx.max_velocity_iteration_count = t["solver_velocity_iters"]

    # Contact parameters
    sim_cfg.physx.bounce_threshold_velocity = t["bounce_threshold_velocity"]
    sim_cfg.physx.friction_offset_threshold = t["friction_offset_threshold"]
    sim_cfg.physx.friction_correlation_distance = t["friction_correlation_distance"]

    # Robot actuator gains
    scene_cfg.robot.actuators["legs"].stiffness = t["joint_stiffness"]
    scene_cfg.robot.actuators["legs"].damping = t["joint_damping"]
    scene_cfg.robot.actuators["legs"].effort_limit_sim = t["effort_limit"]

    print("[physics_tuning] Applied PhysX tuning:")
    print(f"  Solver iters: pos={t['solver_position_iters']}, vel={t['solver_velocity_iters']}")
    print(f"  Joint drive: kp={t['joint_stiffness']}, kv={t['joint_damping']}")
    print(f"  Foot friction: static={t['foot_static_friction']}, dynamic={t['foot_dynamic_friction']}")


def get_ground_material_cfg():
    """Return Isaac Lab PhysicsMaterialCfg for the ground plane.

    Usage in SpotSceneCfg:
        from omni_spot.physics_tuning import get_ground_material_cfg
        ground = AssetBaseCfg(
            ...,
            spawn=sim_utils.GroundPlaneCfg(
                size=(20.0, 20.0),
                physics_material=get_ground_material_cfg(),
            ),
        )
    """
    try:
        try:
            import isaaclab.sim as sim_utils
        except ImportError:
            import omni.isaac.lab.sim as sim_utils
        t = PHYSX_TUNING
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=t["ground_static_friction"],
            dynamic_friction=t["ground_dynamic_friction"],
            restitution=t["ground_restitution"],
        )
    except ImportError:
        return None


def get_foot_material_cfg():
    """Return Isaac Lab PhysicsMaterialCfg for Spot's foot spheres.

    This should be applied as a physics material override on the foot
    collision geoms after USD import. Example:
        from omni_spot.physics_tuning import get_foot_material_cfg
        # In post-import script:
        for foot in ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]:
            apply_material(foot_prim_path, get_foot_material_cfg())
    """
    try:
        try:
            import isaaclab.sim as sim_utils
        except ImportError:
            import omni.isaac.lab.sim as sim_utils
        t = PHYSX_TUNING
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=t["foot_static_friction"],
            dynamic_friction=t["foot_dynamic_friction"],
            restitution=t["foot_restitution"],
        )
    except ImportError:
        return None


def print_tuning_guide():
    """Print the full tuning procedure to stdout."""
    print(TUNING_PROCEDURE)
    print("\nCurrent PHYSX_TUNING values:")
    for k, v in PHYSX_TUNING.items():
        print(f"  {k}: {v}")
