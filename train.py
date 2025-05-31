import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, TOTAL_TIMESTEPS, MODEL_PATH, N_ENVS

# Ensure model path exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Create SB3-compatible env directly (Gymnasium-style works now!)
env = make_vec_env(
    lambda: ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS),
    n_envs=N_ENVS
)

# Define PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,  # CHANGE FROM 0.0001 to 0.0003
    gamma=0.95,
    clip_range=0.1, 
    batch_size=128,      # Larger batch
    n_steps=1024,        # More steps
    ent_coef=0.05,       # Higher entropy for more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128, 64]),  # Larger network
)

# Train
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")