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


# def calculate_training_params(grid_size, num_objects):
#     """Calculate training parameters for curriculum learning"""
#     complexity = grid_size * grid_size * num_objects

#     # Scale timesteps
#     base_timesteps = 150_000
#     timesteps = int(base_timesteps * (1 + complexity / 50))
#     timesteps = min(timesteps, 2_000_000)

#     # Scale max steps
#     max_steps = grid_size * grid_size * 4 + num_objects * 50

#     # Adaptive learning rate
#     if complexity < 30:
#         learning_rate = 3e-4
#     elif complexity < 80:
#         learning_rate = 2e-4
#     else:
#         learning_rate = 1e-4

#     # Adaptive network
#     if complexity < 50:
#         net_arch = [128, 128]
#     elif complexity < 100:
#         net_arch = [256, 256]
#     else:
#         net_arch = [512, 256]

#     # Adaptive exploration
#     if complexity < 30:
#         ent_coef = 0.1
#     elif complexity < 80:
#         ent_coef = 0.05
#     else:
#         ent_coef = 0.02

#     return timesteps, max_steps, learning_rate, net_arch, ent_coef


def calculate_training_params(grid_size, num_objects):
    """Calculate training parameters for curriculum learning - FIXED VERSION"""
    complexity = grid_size * grid_size * num_objects

    # Scale timesteps - KEY FIX: More training time for simple problems
    if complexity <= 16:  # Simple problems like 4x4 with 1-4 objects
        base_timesteps = 200_000  # INCREASED from 150_000
        learning_rate = 3e-4  # Keep standard rate
        net_arch = [64, 64]  # Small network
        ent_coef = 0.05  # Low exploration
    elif complexity <= 50:  # Medium problems
        base_timesteps = 300_000
        learning_rate = 3e-4
        net_arch = [128, 128]
        ent_coef = 0.05
    else:  # Complex problems
        base_timesteps = 500_000
        learning_rate = 2e-4
        net_arch = [256, 256]
        ent_coef = 0.05

    timesteps = min(base_timesteps, 2_000_000)

    # Scale max steps
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    return timesteps, max_steps, learning_rate, net_arch, ent_coef


def get_fixed_ppo_params(grid_size, num_objects):
    """Get the winning PPO hyperparameters that achieve 95% success"""
    complexity = grid_size * grid_size * num_objects

    if complexity <= 16:  # Simple problems - these are the WINNING params!
        return {
            "n_steps": 2048,  # Good rollout length
            "batch_size": 64,  # CRITICAL: Small batch size
            "n_epochs": 10,  # Standard epochs work fine
            "gamma": 0.99,  # Standard discount
            "gae_lambda": 0.95,  # Standard GAE
            "clip_range": 0.2,  # Standard clip range
            "vf_coef": 0.5,  # Standard value function weight
            "max_grad_norm": 0.5,  # Standard gradient clipping
        }
    elif complexity <= 50:
        # Scale up slightly for medium problems
        return {
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    else:
        # Scale up more for complex problems
        return {
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
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
