"""
Quick debug script to see what's happening with the trained model
"""

from stable_baselines3 import PPO
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MODEL_PATH

print("🔍 DEBUGGING TRAINED MODEL")
print("=" * 40)

# Load model
try:
    model = PPO.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test single episode with detailed logging
env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS)
print(f"✅ Environment created: {GRID_SIZE}x{GRID_SIZE}, {NUM_OBJECTS} objects")

obs, _ = env.reset()
print(f"\n🎮 EPISODE DEBUG:")
print(f"Initial state:")
print(f"  Agent at: {env.agent_pos}")
print(f"  Objects: {env.objects}")
print(f"  Carrying: {env.carried_object}")

total_reward = 0
action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Pickup", 5: "Drop"}

for step in range(50):  # Max 50 steps for debug
    # Predict action
    action, _ = model.predict(obs, deterministic=False)
    action_name = action_names.get(int(action), f"Unknown({action})")
    
    # Take step
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    print(f"Step {step+1:2d}: {action_name:6s} → Reward: {reward:6.2f} → Agent: {env.agent_pos} → Carrying: {env.carried_object}")
    
    # Check for completion
    if terminated:
        print(f"\n🎯 TASK COMPLETED at step {step+1}!")
        print(f"💰 Total reward: {total_reward:.2f}")
        print("✅ SUCCESS!")
        break
    elif truncated:
        print(f"\n⏰ TIMEOUT at step {step+1}")
        print(f"💰 Total reward: {total_reward:.2f}")
        print("❌ FAILED!")
        break

if not terminated and not truncated:
    print(f"\n🔄 Episode still running after 50 steps")
    print(f"💰 Current reward: {total_reward:.2f}")
    print("⚠️  Needs more steps or better strategy")

# Check environment state
print(f"\n📊 FINAL STATE:")
print(f"  Agent at: {env.agent_pos}")
print(f"  Objects: {env.objects}")
print(f"  Carrying: {env.carried_object}")
print(f"  Task complete: {env.is_episode_complete()}")

# Check if objects are in correct zones
if env.objects:
    print(f"\n🎯 OBJECT ANALYSIS:")
    for pos, obj_type in env.objects.items():
        correct = env.is_correct_zone(pos, obj_type)
        zone = "red" if pos[1] < GRID_SIZE // 2 else "blue"
        obj_name = "red" if obj_type == 1 else "blue"
        status = "✅" if correct else "❌"
        print(f"  {status} {obj_name} object at {pos} in {zone} zone")

print("\n🎯 Debug completed!")