"""
Video Recorder for Isaac Lab Spot Environment
===============================================
Records RGB frames from an overhead + chase camera during evaluation
episodes and writes them to MP4 using imageio.

Usage:
    from omni_spot.video_recorder import record_episodes
    record_episodes(env, trainer, n_episodes=5, output_path="spot_eval.mp4")
"""

from __future__ import annotations

import torch
import numpy as np

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio as iio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False


def record_episodes(
    env,
    trainer,
    n_episodes: int = 5,
    output_path: str = "spot_eval.mp4",
    max_steps: int = 500,
    fps: int = 25,
    camera_name: str = "video_cam",
    resolution: tuple[int, int] = (480, 640),
):
    """Record evaluation episodes to MP4.

    Parameters
    ----------
    env : NavEnv
        Isaac Lab environment (must have a camera named `camera_name` in scene,
        OR falls back to env.render() if available).
    trainer : PPOTrainer
        Trained PPO agent.
    n_episodes : int
        Number of episodes to record.
    output_path : str
        Path to save MP4 file.
    max_steps : int
        Max steps per episode (safety cutoff).
    fps : int
        Video frame rate.
    camera_name : str
        Name of the RGB camera in the scene for recording.
    resolution : tuple
        (height, width) for the recording camera.
    """
    if not HAS_IMAGEIO:
        print("[video] ERROR: imageio not installed. Run: pip install imageio[ffmpeg]")
        return

    print(f"[video] Recording {n_episodes} episodes to {output_path}")
    frames = []

    trainer.net.eval()
    obs, _ = env.reset()

    episodes_done = 0
    step = 0

    with torch.no_grad():
        while episodes_done < n_episodes and step < n_episodes * max_steps:
            # Get action from policy
            action, _, _ = trainer.sample_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            step += 1

            # Capture frame
            frame = _capture_frame(env, camera_name)
            if frame is not None:
                frames.append(frame)

            # Count completed episodes (any env finishing counts)
            n_done = done.any().sum().item() if done.dim() > 0 else int(done.item())
            if done.any():
                episodes_done += 1
                if episodes_done < n_episodes:
                    # Add a few blank frames as episode separator
                    if frames:
                        for _ in range(fps // 2):  # 0.5s gap
                            frames.append(frames[-1])

    if not frames:
        print("[video] No frames captured. Check camera setup.")
        return

    print(f"[video] Writing {len(frames)} frames at {fps} fps...")
    iio.imwrite(output_path, np.stack(frames), fps=fps, codec="h264")
    print(f"[video] Saved to {output_path}")
    return output_path


def _capture_frame(env, camera_name: str) -> np.ndarray | None:
    """Try to capture an RGB frame from the environment.

    Tries multiple approaches:
    1. Named RGB camera in the scene (if added to SpotSceneCfg)
    2. env.render() (Isaac Lab viewport render)
    3. Isaac Sim viewport API
    """
    # Approach 1: Named camera in scene
    try:
        cam = env.scene[camera_name]
        rgb = cam.data.output["rgb"]  # (num_envs, H, W, 3)
        # Take first env
        frame = rgb[0].cpu().numpy().astype(np.uint8)
        return frame
    except (KeyError, AttributeError, RuntimeError):
        pass

    # Approach 2: env.render() — some Isaac Lab envs support this
    try:
        frame = env.render(mode="rgb_array")
        if isinstance(frame, np.ndarray):
            return frame
        if isinstance(frame, torch.Tensor):
            return frame.cpu().numpy().astype(np.uint8)
    except (TypeError, AttributeError, NotImplementedError):
        pass

    # Approach 3: Isaac Sim viewport capture
    try:
        from omni.isaac.core.utils.viewports import get_viewport_data
        data = get_viewport_data()
        if data is not None:
            return np.array(data, dtype=np.uint8)
    except ImportError:
        pass

    return None


def record_from_viewport(
    output_path: str = "spot_viewport.mp4",
    n_frames: int = 300,
    fps: int = 30,
):
    """Record frames directly from Isaac Sim's active viewport.

    Useful for getting a rendered view of Spot without a dedicated camera.
    Must be called while the simulation is running.
    """
    if not HAS_IMAGEIO:
        print("[video] ERROR: imageio not installed.")
        return

    try:
        from omni.isaac.sensor import Camera as ViewportCamera
        from omni.isaac.core.utils.stage import get_current_stage
    except ImportError:
        print("[video] ERROR: Must run inside Isaac Sim runtime.")
        return

    print(f"[video] Capturing {n_frames} viewport frames...")
    frames = []

    for i in range(n_frames):
        try:
            from omni.isaac.core.utils.viewports import get_viewport_data
            frame = get_viewport_data()
            if frame is not None:
                frames.append(np.array(frame, dtype=np.uint8))
        except Exception:
            pass

    if frames:
        iio.imwrite(output_path, np.stack(frames), fps=fps, codec="h264")
        print(f"[video] Saved viewport recording to {output_path}")
    else:
        print("[video] No viewport frames captured.")
