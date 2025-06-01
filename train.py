import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, TOTAL_TIMESTEPS, MODEL_PATH, N_ENVS


os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


env = make_vec_env(
    lambda: ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS),
    n_envs=N_ENVS
)

# Define PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    clip_range=0.2,
    batch_size=256,
    n_steps=2048,
    ent_coef=0.2,                    
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
)

# Train
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")