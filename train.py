import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MAX_STEPS, MODEL_PATH

# Ensure model path exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Create environment
def make_env():
    return ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, max_steps=MAX_STEPS)


env = make_vec_env(make_env, n_envs=4)

print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
print(f"Number of objects: {NUM_OBJECTS}")


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    clip_range=0.2,
    batch_size=256,
    n_steps=2048,
    ent_coef=0.1,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
    device="cpu",
)

# Create evaluation environment
eval_env = make_vec_env(make_env, n_envs=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{MODEL_PATH}_best",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# Train
model.learn(total_timesteps=200_000, callback=eval_callback)
model.save(MODEL_PATH)

print(f"\n✅ Model saved to: {MODEL_PATH}")
