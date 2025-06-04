# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

# Environment Settings
GRID_SIZE = 4
NUM_OBJECTS = 1
USE_SHAPING = True

# Training Settings
TIMESTEPS = 200_000
LEARNING_RATE = 3e-4
N_ENVS = 4

# Evaluation Settings
EVAL_EPISODES = 15

# Video Settings
VIDEO_EPISODES = 3
VIDEO_FORMAT = "gif"  # "gif" or "mp4"
VIDEO_DURATION = 1.0  # seconds per frame (higher = slower)
VIDEO_FPS = 1  # frames per second for mp4

# Paths
MODEL_NAME = "zoning_agent"
MODELS_DIR = "models"
LOGS_DIR = "logs"
RECORDINGS_DIR = "recordings"

# =============================================================================
# DERIVED SETTINGS (automatically calculated - don't edit)
# =============================================================================

# Calculate max steps based on grid size and objects
MAX_STEPS = GRID_SIZE * GRID_SIZE * 4 + NUM_OBJECTS * 50

# Full paths
MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}"
LOG_PATH = f"{LOGS_DIR}/{MODEL_NAME}"

# Auto-scale some parameters based on complexity
COMPLEXITY = GRID_SIZE * GRID_SIZE * NUM_OBJECTS

if COMPLEXITY < 30:
    NET_ARCH = [128, 128]
    ENT_COEF = 0.1
elif COMPLEXITY < 80:
    NET_ARCH = [256, 256]
    ENT_COEF = 0.05
else:
    NET_ARCH = [512, 256]
    ENT_COEF = 0.02

# if COMPLEXITY < 20:
#     # Small problems - compact network
#     NET_ARCH = [256, 256, 128]
# elif COMPLEXITY < 50:
#     # Medium problems - deeper network
#     NET_ARCH = [512, 512, 256, 128]
# elif COMPLEXITY < 100:
#     # Large problems - much deeper network
#     NET_ARCH = [1024, 512, 512, 256, 128]
# else:
#     # Very large problems - deep network with residual-like structure
#     NET_ARCH = [1024, 1024, 512, 512, 256, 256, 128]

# Curriculum settings (for grid_size_curriculum.py)
# CURRICULUM_GRID_SIZES = [4, 5, 6, 7, 8, 9, 10, 12]
# CURRICULUM_MAX_OBJECTS = {4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 12: 10}

CURRICULUM_GRID_SIZES = [4, 5]
CURRICULUM_MAX_OBJECTS = {4: 2, 5: 2}

# =============================================================================
# QUICK CONFIGURATION PRESETS
# =============================================================================


def use_small_config():
    """Quick preset for small/fast testing"""
    global GRID_SIZE, NUM_OBJECTS, TIMESTEPS, MAX_STEPS
    GRID_SIZE = 4
    NUM_OBJECTS = 1
    TIMESTEPS = 50_000
    MAX_STEPS = GRID_SIZE * GRID_SIZE * 4 + NUM_OBJECTS * 50


def use_medium_config():
    """Quick preset for medium complexity"""
    global GRID_SIZE, NUM_OBJECTS, TIMESTEPS, MAX_STEPS
    GRID_SIZE = 6
    NUM_OBJECTS = 2
    TIMESTEPS = 200_000
    MAX_STEPS = GRID_SIZE * GRID_SIZE * 4 + NUM_OBJECTS * 50


def use_large_config():
    """Quick preset for large/challenging"""
    global GRID_SIZE, NUM_OBJECTS, TIMESTEPS, MAX_STEPS
    GRID_SIZE = 8
    NUM_OBJECTS = 3
    TIMESTEPS = 500_000
    MAX_STEPS = GRID_SIZE * GRID_SIZE * 4 + NUM_OBJECTS * 50


def print_config():
    """Print current configuration"""
    print("🔧 Current Configuration:")
    print(f"   Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"   Objects: {NUM_OBJECTS}")
    print(f"   Max Steps: {MAX_STEPS}")
    print(f"   Timesteps: {TIMESTEPS:,}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Network: {NET_ARCH}")
    print(f"   Model Path: {MODEL_PATH}")
    print()


# Apply default configuration
if __name__ == "__main__":
    print_config()
