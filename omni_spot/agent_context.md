Task: Generate verification tests + smoke test script for the repo
Deliver two copyable scripts (no git push):

verify_setup.sh — checks all installed components are working
smoke_test.sh — runs the actual repo training pipeline end-to-end

Key facts from codebase

train.py imports SpotNavEnv (Isaac Lab) or exits with error — full Omniverse needed
mock_env.py has MockSpotEnv(num_envs, device) — pure PyTorch, no Omniverse needed
mock_env.py API: env.reset() → (obs, info), env.step(action) → (obs, reward, terminated, truncated, info)
ppo.py has PPOTrainer(n_envs, n_steps, lr) + collect_rollout(env, obs) + update(batch)
spot_actor_critic.py has the network
Train script writes logs to omni_logs/<timestamp>/train_log.csv
omni_spot is not installed — must run with PYTHONPATH=~/claude_code
MJCF→USD conversion is still unresolved (try isaaclab.sim.converters)

Verification tests to include

nvidia-smi — GPU present
python -c "import torch; torch.cuda.is_available()" — CUDA works
python -c "import isaaclab" — IsaacLab importable
python -c "import isaacsim" — Isaac Sim importable
python -c "import ray" — ray present
python -c "import gymnasium" — gymnasium present
Mock env instantiation — no Omniverse needed, validates Python pipeline
PPO trainer + mock env 2-step rollout — validates training loop
USD file exists check
DCV server running check

Smoke test levels

Level 1 (no Omniverse): MockSpotEnv + PPOTrainer for 5 updates — validates network, PPO, rewards
Level 2 (full): PYTHONPATH=~/claude_code python -m omni_spot.train --num_envs 64 --n_steps 128 --total_updates 5 — requires USD file + Isaac Lab


Project Overview

Repo: https://github.com/Raj9408612613/claude_code.git
Branch: claude/review-isaac-lab-compatibility-Q1vx1
Goal: Train a Spot robot in Isaac Lab (PhysX 5 GPU physics) using PPO
Key directory: omni_spot/ — contains training code, env config, MJCF→USD converter

omni_spot/ structure
omni_spot/
├── __init__.py
├── config.py                  # Hyperparameters & constants
├── convert_mjcf_to_usd.py    # MJCF→USD converter (needs Omniverse Kit runtime)
├── train.py                   # Training entry point (Isaac Lab DirectRLEnv)
├── spot_env.py                # DirectRLEnv implementation
├── spot_env_cfg.py            # Isaac Lab environment config
├── spot_actor_critic.py       # PyTorch Actor-Critic network
├── ppo.py                     # PyTorch PPO trainer
├── reward.py                  # Reward computation
├── diagnostics.py             # Training diagnostics
├── video_recorder.py          # Video recording utility
├── physics_tuning.py          # Physics parameter tuning
└── mock_env.py                # Pure PyTorch mock environment (no Omniverse needed)
Model files (in repo root)
models/
├── spot_scene.xml             # MJCF model of Spot robot
└── assets/                    # 23 OBJ mesh files for Spot
    ├── body_0.obj, body_1.obj, body_collision.obj
    ├── front_left_hip.obj, front_right_hip.obj, ...
    ├── *_upper_leg_*.obj, *_lower_leg_*.obj
    └── *_collision.obj

No spot_scene.usd exists yet — must be generated via MJCF conversion
The converter (convert_mjcf_to_usd.py) requires full Omniverse Kit runtime (omni.kit.app, omni.importer.mjcf)
Alternative: Use isaaclab.sim.converters.MjcfConverter / MjcfConverterCfg — this may work without the full Kit runtime (untested, was the next step)


All Known Issues & Fixes (Discovered Through Debugging)
1. Python Version Incompatibility

Problem: Conda defaults to Python 3.13. Isaac Sim has NO wheels for 3.13.
Isaac Sim version support:

4.x → Python 3.10 only
5.x → Python 3.11 only
6.0 → Python 3.12 only


Fix: Create a conda env with Python 3.11 (matches Isaac Sim 5.1.0 and the rl_games@python3.11 fork branch)

bashconda create -n isaaclab python=3.11 -y
conda activate isaaclab
2. Conda Terms of Service

Problem: conda create fails with CondaToSNonInteractiveError if ToS not accepted
Fix: Accept before creating env

bashconda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
3. pip resolution-too-deep Error

Problem: ./isaaclab.sh --install triggers pip dependency resolution explosion. Root cause: rl-games (from git+https://github.com/isaac-sim/rl_games.git@python3.11) requires ray>=2.45.0,<3.0.0, and ray has a massive transitive dependency tree that creates combinatorial explosion with IsaacLab's other deps.
Fix: Pre-install ray, then use legacy resolver for rl_games

bashpip install "ray[default]==2.45.0"
pip install --use-deprecated=legacy-resolver \
    "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
4. setuptools Flat-Layout Error

Problem: setuptools>=75 rejects IsaacLab's multi-package root layout (apps, source, docker discovered as top-level packages)
Fix: Pin setuptools before install

bashpip install "setuptools<75.0.0"
5. dex-retargeting==0.4.6 Not Available

Problem: pip install -e "source/isaaclab" fails because dex-retargeting==0.4.6 has no wheel for Python 3.11/3.13. It's for dexterous hand manipulation — NOT needed for Spot robot.
Fix: Install isaaclab with --no-deps, then manually install needed deps

bashpip install --no-deps -e "source/isaaclab"
pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict
6. Missing toml Module

Problem: import isaaclab fails with ModuleNotFoundError: No module named 'toml' because --no-deps skipped it
Fix: pip install toml (included in the manual dep install above)

7. IsaacLab Source Layout

Path: ~/IsaacLab/source/
Packages (install in this order):

source/isaaclab — core (use --no-deps)
source/isaaclab_assets — asset configs
source/isaaclab_tasks — task definitions
source/isaaclab_rl — RL wrappers
source/isaaclab_contrib — community contributions (optional)
source/isaaclab_mimic — mimicry tools (optional)



8. MJCF→USD Conversion

Problem: convert_mjcf_to_usd.py requires full Omniverse Kit runtime (omni.kit.app), which is NOT included in pip-installed Isaac Sim
Attempted fix: Use IsaacLab's built-in converter instead:

pythonfrom isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
cfg = MjcfConverterCfg(
    asset_path='models/spot_scene.xml',
    usd_dir='models',
    usd_file_name='spot_scene.usd',
    fix_base=False,
    import_sites=True,
    self_collision=False,
)
converter = MjcfConverter(cfg)

Status: UNTESTED — this was the next step when the session ended. If this also requires the Kit runtime, alternatives are:

Install Isaac Sim via Omniverse Launcher (not pip) to get the full Kit runtime
Pre-generate the USD on a machine with full Isaac Sim and commit it to the repo
Modify spot_env_cfg.py to load MJCF directly if IsaacLab supports it



9. omni_spot Module Not Found

Problem: python -m omni_spot.train fails because omni_spot is not an installed package
Fix: Set PYTHONPATH to the repo root

bashcd ~/claude_code
PYTHONPATH=. python -m omni_spot.train --num_envs 64 --n_steps 128 --total_updates 5

Or from IsaacLab:

bashPYTHONPATH=~/claude_code ./isaaclab.sh -p -m omni_spot.train ...
10. pip Dependency Conflict Warnings (NOT Errors)

After installing with --no-deps, pip shows ERROR: pip's dependency resolver does not currently take into account... warnings about missing optional deps like dex-retargeting, transformers, pytest, hidapi, etc.
These are WARNINGS, not errors. The install succeeds. These packages are not needed for Spot training.


Verified Working Install Order (Python 3.11, Isaac Sim 5.1.0)
bash# 1. Conda env
conda create -n isaaclab python=3.11 -y && conda activate isaaclab

# 2. Isaac Sim
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk

# 3. IsaacLab pre-reqs
pip install "ray[default]==2.45.0"
pip install "setuptools<75.0.0"
pip install --use-deprecated=legacy-resolver \
    "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"

# 4. IsaacLab core
cd ~/IsaacLab
pip install --no-deps -e "source/isaaclab"
pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict

# 5. IsaacLab extensions
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_assets"
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_tasks"
pip install --use-deprecated=legacy-resolver -e "source/isaaclab_rl"
pip install tensorboard "imageio[ffmpeg]"

# 6. Verification
python -c "import isaaclab; print('IsaacLab OK')"
python -c "import isaacsim; print('Isaac Sim OK')"
All above steps verified working. The remaining unresolved step is MJCF→USD conversion.

EC2 Instance Details

OS: Ubuntu 22.04
GPU: NVIDIA (user has a Deep Learning AMI with drivers pre-installed)
Remote Desktop: NICE DCV on port 8443
Conda: Miniconda3 at ~/miniconda3
IsaacLab: Cloned at ~/IsaacLab
Project repo: Cloned at ~/claude_code

NICE DCV Setup
bashsudo apt install -y ubuntu-desktop gdm3
wget https://d1uj6qtbmh3dt5.cloudfront.net/nice-dcv-ubuntu2204-x86_64.tgz
tar -xzf nice-dcv-ubuntu2204-x86_64.tgz && cd nice-dcv-*-x86_64/
sudo apt install -y ./nice-dcv-server_*.deb ./nice-dcv-web-viewer_*.deb ./nice-xdcv_*.deb
sudo systemctl enable dcvserver && sudo systemctl start dcvserver
sudo dcv create-session --owner ubuntu --type virtual my-session
sudo passwd ubuntu
# Open port 8443 in EC2 Security Group → https://<public-ip>:8443

Smoke Test Command
bashconda activate isaaclab
cd ~/claude_code
PYTHONPATH=. python -m omni_spot.train --num_envs 64 --n_steps 128 --total_updates 5
What's Left / Next Steps

Resolve MJCF→USD conversion — try isaaclab.sim.converters.MjcfConverter or install full Omniverse Kit
Run smoke test — verify training loop works end-to-end
Create scripts/setup_ec2_isaac.sh — idempotent setup script incorporating all fixes above
