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


def calculate_training_params(grid_size, num_objects):
    """Calculate training parameters for curriculum learning"""
    complexity = grid_size * grid_size * num_objects

    # Scale timesteps
    base_timesteps = 150_000
    timesteps = int(base_timesteps * (1 + complexity / 50))
    timesteps = min(timesteps, 2_000_000)

    # Scale max steps
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    # Adaptive learning rate
    if complexity < 30:
        learning_rate = 3e-4
    elif complexity < 80:
        learning_rate = 2e-4
    else:
        learning_rate = 1e-4

    # Adaptive network
    if complexity < 50:
        net_arch = [128, 128]
    elif complexity < 100:
        net_arch = [256, 256]
    else:
        net_arch = [512, 256]

    # Adaptive exploration
    if complexity < 30:
        ent_coef = 0.1
    elif complexity < 80:
        ent_coef = 0.05
    else:
        ent_coef = 0.02

    return timesteps, max_steps, learning_rate, net_arch, ent_coef


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
