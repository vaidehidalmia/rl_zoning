import time
import numpy as np
from stable_baselines3 import PPO
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MODEL_PATH

# Action label mapping
ACTION_MEANINGS = {
    0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Pickup", 5: "Drop"
}

print("🔍 DEBUGGING EVALUATION")
print("=" * 50)

# Debug 1: Check config values
print(f"Config GRID_SIZE: {GRID_SIZE}")
print(f"Config NUM_OBJECTS: {NUM_OBJECTS}")
print(f"Model path: {MODEL_PATH}")

# Debug 2: Load and inspect model
print("\n📦 Loading model...")
model = PPO.load(MODEL_PATH)
print(f"✅ Model loaded successfully")
print(f"Model training timesteps: {model.num_timesteps}")
print(f"Model policy: {type(model.policy)}")

# Debug 3: Create and inspect environment
print("\n🌍 Creating environment...")
env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode="human")
print(f"Environment grid_size: {env.grid_size}")
print(f"Environment num_objects: {env.num_objects}")
print(f"Environment action_space: {env.action_space}")
print(f"Environment observation_space: {env.observation_space}")

# Debug 4: Reset and inspect initial state
print("\n🔄 Resetting environment...")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Agent position: {env.agent_pos}")
print(f"Carried object: {env.carried_object}")
print(f"Objects on grid: {env.objects}")

# Debug 5: Test action predictions
print("\n🎯 Testing action predictions...")
for test_mode in [True, False]:
    action, _ = model.predict(obs, deterministic=test_mode)
    action_int = int(action)
    mode_name = "deterministic" if test_mode else "stochastic"
    print(f"Action ({mode_name}): {action_int} ({ACTION_MEANINGS[action_int]})")

# Debug 6: Test action distribution over multiple predictions
print("\n📊 Action distribution test (10 predictions):")
action_counts = {i: 0 for i in range(6)}
for _ in range(10):
    action, _ = model.predict(obs, deterministic=False)
    action_counts[int(action)] += 1

for action_id, count in action_counts.items():
    print(f"  {ACTION_MEANINGS[action_id]}: {count}/10")

# Debug 7: Check if model is stuck
if action_counts[0] >= 8:  # More than 80% Up actions
    print("⚠️  MODEL APPEARS STUCK ON UP ACTION!")
    print("💡 Trying non-deterministic evaluation...")
    use_deterministic = False
else:
    print("✅ Model shows action variety")
    use_deterministic = True

print("\n" + "=" * 50)
print("🎮 Starting evaluation...")
print(f"Using deterministic: {use_deterministic}")
print("=" * 50)

# Main evaluation loop with enhanced debugging
obs, _ = env.reset()
total_reward = 0
action_history = []

for step in range(100):
    env.render()
    
    # Get action with chosen mode
    action, _ = model.predict(obs, deterministic=use_deterministic)
    action_int = int(action)
    action_history.append(action_int)
    
    # Show environment state before action
    if step < 10 or step % 20 == 0:  # Detailed info for first 10 steps and every 20th
        print(f"\n--- Step {step + 1} Details ---")
        print(f"Agent at: {env.agent_pos}")
        print(f"Carrying: {env.carried_object}")
        print(f"Objects: {env.objects}")
        print(f"Action: {action_int} ({ACTION_MEANINGS[action_int]})")
    
    # Take action
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    action_name = ACTION_MEANINGS.get(action_int, "Unknown")
    print(f"Step {step + 1} | Action: {action_int} ({action_name}) | Reward: {reward:.3f}")
    
    # Check for loops
    if len(action_history) >= 10:
        recent_actions = action_history[-10:]
        if len(set(recent_actions)) == 1:
            print(f"⚠️  STUCK IN LOOP: Last 10 actions all {ACTION_MEANINGS[recent_actions[0]]}")
            print("🔄 Forcing random action...")
            action = np.random.randint(0, 6)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Forced action: {action} ({ACTION_MEANINGS[action]}) | Reward: {reward:.3f}")
            action_history.append(action)
    
    time.sleep(0.25)
    
    if terminated or truncated:
        env.render()
        print(f"\n🏁 Episode finished at step {step + 1}")
        if terminated:
            print("✅ Task completed successfully!")
        else:
            print("⏰ Episode truncated (timeout)")
        break

print(f"\n🎯 Total Reward: {total_reward:.2f}")

# Final analysis
print(f"\n📈 Action Distribution in Episode:")
episode_action_counts = {i: action_history.count(i) for i in range(6)}
for action_id, count in episode_action_counts.items():
    percentage = (count / len(action_history)) * 100
    print(f"  {ACTION_MEANINGS[action_id]}: {count} ({percentage:.1f}%)")

if episode_action_counts[4] == 0 and episode_action_counts[5] == 0:
    print("❌ Agent never tried pickup or drop actions!")
    print("💡 This suggests the model learned a degenerate policy")