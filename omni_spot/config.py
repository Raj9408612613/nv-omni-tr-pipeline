"""
Environment & Training Configuration
======================================
Ported from config.py + jax_ppo.py constants.
All hyperparameters preserved exactly from the JAX version.
"""

# ── PPO Hyperparameters ──────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENT_COEF     = 0.01
VF_COEF      = 1.0
MAX_GRAD     = 1.5
LR           = 3e-4
N_EPOCHS     = 8
MINIBATCH_SZ = 512
TARGET_KL    = 0.03

# ── Network dimensions ──────────────────────────────────────────────────────
CNN_FEAT_DIM = 256
PROPRIO_DIM  = 37
ACTION_DIM   = 12
LOG_STD_MIN  = -5.0
LOG_STD_MAX  =  2.0

# ── Reward weights (from jax_reward.py) ──────────────────────────────────────
GOAL_BONUS       =   200.0
GOAL_TOL         =     0.5    # metres
PROGRESS_W       =    50.0
COLLISION_PEN    =   -10.0
NEAR_COLL_PEN    =    -2.0
NEAR_COLL_THRESH =    0.35   # metres
UPRIGHT_W        =    -0.3
HEIGHT_W         =    -1.0
TARGET_HEIGHT    =    0.5   # real Spot standing height (m)
ENERGY_W         =  -0.005
SMOOTH_W         =  -0.002
ALIVE_BONUS      =     0.5
HEADING_W        =     0.3
# Velocity tracking toward goal
VEL_TRACK_W = 1.0    # reward per (m/s) of goal-directed velocity
VEL_TRACK_CAP = 1.5  # m/s — saturation speed, reward doesn't grow beyond this

# ── Robot physical properties (real Boston Dynamics Spot) ────────────────────
SPOT_MASS         = 32.7                    # kg (BD datasheet)
SPOT_BODY_DIMS    = [1.1, 0.5, 0.191]      # L×W×H in metres (1100×500×191 mm)

# ── Physics ──────────────────────────────────────────────────────────────────
PHYSICS_DT        = 0.005    # PhysX sim timestep (200 Hz)
CONTROL_DT        = 0.02     # RL control rate (50 Hz) = 4 substeps
PHYSICS_SUBSTEPS  = 4

# ── Joint limits (real Spot URDF values) ─────────────────────────────────────
#   hip_x (abduction):  ±0.785398 rad
#   hip_y (flexion):    -0.89012 / +2.29511 rad
#   knee:               per-leg upper from URDF (all close to -0.254)
JOINT_LOWER = [
    -0.785398, -0.89012, -2.7929,   # fl: hx, hy, kn
    -0.785398, -0.89012, -2.7929,   # fr
    -0.785398, -0.89012, -2.7929,   # hl
    -0.785398, -0.89012, -2.7929,   # hr
]
JOINT_UPPER = [
     0.785398,  2.29511,  -0.254402, # fl
     0.785398,  2.29511,  -0.255648, # fr  (was 2.24363 — corrected to match URDF)
     0.785398,  2.29511,  -0.247067, # hl
     0.785398,  2.29511,  -0.248282, # hr
]

# ── Joint velocity limits (rad/s) ─────────────────────────────────────────────
JOINT_VEL_LIMITS = [
    4.0, 4.0, 6.0,   # fl: hx, hy, kn  (hip: 4 rad/s, knee: 6 rad/s)
    4.0, 4.0, 6.0,   # fr
    4.0, 4.0, 6.0,   # hl
    4.0, 4.0, 6.0,   # hr
]

STANDING_POSE = [0.0, 1.04, -1.8] * 4  # home keyframe (12 joints)

# ── Terrain / room ───────────────────────────────────────────────────────────
PATCH_SIZE = 8.0     # each terrain patch is 8 × 8 m
PATCH_HALF = 3.5     # usable half-width inside patch (0.5 m border)
ROOM_HALF  = PATCH_HALF  # backward-compat alias (mock env, reward clamps)

TERRAIN_ROWS = 4     # 4 difficulty rows (row 0 = easiest)
TERRAIN_COLS = 4     # 4 columns per row  → 16 patch templates
FLAT_TERRAIN_ROW_MAX = 1  # rows 0 & 1 are flat → place obstacles there

# ── Camera ───────────────────────────────────────────────────────────────────
N_CAMS   = 3
CAM_H    = 120
CAM_W    = 160
H_FOV    = 87.0    # degrees
V_FOV    = 58.0    # degrees
MIN_DEPTH = 0.1
MAX_DEPTH = 10.0

# ── Obstacles ────────────────────────────────────────────────────────────────
# Reduced to 2 static boxes on flat-terrain patches only.
# Humanoid entry kept in the array (index 2) but disabled; its slot keeps
# HUMANOID_MOCAP_IDX valid so existing array indexing doesn't break.
N_STATIC   = 2
N_DYNAMIC  = 0
N_HUMANOID = 1           # slot kept, always off-scene
N_OBS      = N_STATIC + N_DYNAMIC + N_HUMANOID  # 3
HUMANOID_MOCAP_IDX = N_STATIC + N_DYNAMIC        # = 2

OBS_HALF_SIZES = [
    [0.30, 0.30, 0.50],  # static box 0
    [0.25, 0.40, 0.40],  # static box 1
    [0.30, 0.30, 1.00],  # humanoid AABB (always off-scene)
]

# ── Humanoid walking obstacle ────────────────────────────────────────────────
HUMANOID_OBSTACLE = {
    "enabled":        False,   # disabled — terrain curriculum uses no humanoid
    "speed":          0.8,
    "stride_freq":    1.2,
    "patrol_radius":  1.5,
    "mocap_z":        1.0,
    "wp_switch_dist": 0.2,
}
