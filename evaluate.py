import time
from stable_baselines3 import PPO
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MODEL_PATH

# Action label mapping
ACTION_MEANINGS = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "Pickup",
    5: "Drop"
}

# Load trained model
model = PPO.load(MODEL_PATH)

# Create environment with rendering
env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode="human")

obs, _ = env.reset()
total_reward = 0

print("\n🎮 Starting evaluation...\n")

for step in range(100):
    env.render()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    action_int = int(action) 
    action_name = ACTION_MEANINGS.get(action_int, "Unknown")
    print(f"Step {step + 1} | Action: {action_int} ({action_name}) | Reward: {reward:.2f}")
    time.sleep(0.25)

    if terminated or truncated:
        env.render()
        print(f"\n🏁 Episode finished at step {step + 1}")
        break

print(f"\n🎯 Total Reward: {total_reward:.2f}")