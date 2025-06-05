from config import *
from environment import (
    ZoningEnv,
)


def make_env(render_mode=None, grid_size=None, num_objects=None, use_shaping=None):
    """Create environment with current config settings or custom parameters"""
    return ZoningEnv(
        grid_size=grid_size or GRID_SIZE,
        num_objects=num_objects or NUM_OBJECTS,
        use_shaping=use_shaping if use_shaping is not None else USE_SHAPING,
        render_mode=render_mode,
    )


def get_model_path(grid_size=None, num_objects=None, stage=None):
    """Get model path for specific configuration"""
    if stage is not None:
        return (
            f"{MODELS_DIR}/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}"
        )
    elif grid_size is not None and num_objects is not None:
        return f"{MODELS_DIR}/{MODEL_NAME}_grid_{grid_size}_obj_{num_objects}"
    else:
        return MODEL_PATH


def get_log_path(grid_size=None, num_objects=None, stage=None):
    """Get log path for specific configuration"""
    if stage is not None:
        return (
            f"{LOGS_DIR}/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}/"
        )
    elif grid_size is not None and num_objects is not None:
        return f"{LOGS_DIR}/{MODEL_NAME}_grid_{grid_size}_obj_{num_objects}/"
    else:
        return f"{LOG_PATH}/"


def calculate_training_params_transfer(grid_size, num_objects, has_previous=False):
    """Training params with transfer learning adjustments"""
    complexity = grid_size * grid_size * num_objects

    # Base parameters
    if num_objects == 1:
        if complexity <= 25:
            base_timesteps = 200_000
            learning_rate = 3e-4
            net_arch = [64, 64]
            ent_coef = 0.05
        else:
            base_timesteps = 300_000
            learning_rate = 3e-4
            net_arch = [128, 128]
            ent_coef = 0.05
    else:
        # Multi-object with transfer learning needs less training
        if complexity <= 32:  # 4x4 with 2 objects
            base_timesteps = 600_000 if has_previous else 800_000  # Less with transfer!
            learning_rate = 2e-4
            net_arch = [128, 128]
            ent_coef = 0.08
        else:  # 5x5 with 2+ objects
            base_timesteps = (
                800_000 if has_previous else 1_200_000
            )  # Less with transfer!
            learning_rate = 1e-4
            net_arch = [256, 256]
            ent_coef = 0.1

    timesteps = min(base_timesteps, 2_000_000)
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    return timesteps, max_steps, learning_rate, net_arch, ent_coef


def get_fixed_ppo_params_transfer(grid_size, num_objects):
    """PPO params optimized for transfer learning"""
    complexity = grid_size * grid_size * num_objects

    if num_objects == 1:
        return {
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    else:
        # Multi-object params
        if complexity <= 32:
            return {
                "n_steps": 4096,
                "batch_size": 128,
                "n_epochs": 15,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.1,  # Smaller clip for fine-tuning
                "vf_coef": 1.0,
                "max_grad_norm": 0.5,
            }
        else:
            return {
                "n_steps": 4096,
                "batch_size": 256,
                "n_epochs": 20,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.1,
                "vf_coef": 1.0,
                "max_grad_norm": 0.5,
            }


def load_model_safe(model_path=None):
    """Safely load PPO model with proper device handling"""
    from stable_baselines3 import PPO

    path = model_path or MODEL_PATH
    try:
        model = PPO.load(path, device="cpu")
        return model
    except Exception as e:
        print(f"❌ Could not load model from {path}: {e}")
        return None


def create_directories():
    """Create necessary directories"""
    import os

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
