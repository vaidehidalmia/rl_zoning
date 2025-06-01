from stable_baselines3 import PPO
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MODEL_PATH

# Load trained model
print(f"📦 Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# Create environment 
env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode=None)

print(f"\n🎮 Evaluating agent on {GRID_SIZE}x{GRID_SIZE} grid with {NUM_OBJECTS} object(s)")
print("=" * 60)

# Statistics tracking
total_episodes = 20
successful_episodes = 0
total_steps = 0
episode_rewards = []
episode_lengths = []

# Run multiple episodes to see performance
for episode in range(total_episodes):
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(200):  # Max 200 steps per episode
        action, _ = model.predict(obs, deterministic=False) 
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            episode_lengths.append(step + 1)
            episode_rewards.append(total_reward)
            total_steps += step + 1
            
            if terminated:
                successful_episodes += 1
                print(f"Episode {episode + 1:2d}: ✅ COMPLETED in {step + 1:3d} steps (reward: {total_reward:6.1f})")
            else:
                print(f"Episode {episode + 1:2d}: ❌ TIMEOUT  in {step + 1:3d} steps (reward: {total_reward:6.1f})")
            break

print("\n" + "=" * 60)
print("📊 EVALUATION SUMMARY:")
print(f"Success Rate:     {successful_episodes}/{total_episodes} ({100*successful_episodes/total_episodes:.1f}%)")
print(f"Average Steps:    {sum(episode_lengths)/len(episode_lengths):.1f}")
print(f"Average Reward:   {sum(episode_rewards)/len(episode_rewards):.1f}")

if successful_episodes > 0:
    successful_lengths = [length for i, length in enumerate(episode_lengths) 
                         if episode_rewards[i] > 50]  # Successful episodes
    if successful_lengths:
        print(f"Avg Steps (Success): {sum(successful_lengths)/len(successful_lengths):.1f}")

print(f"Best Episode:     {min(episode_lengths)} steps")
print(f"Worst Episode:    {max(episode_lengths)} steps")

# Performance classification
success_rate = successful_episodes / total_episodes
if success_rate >= 0.9:
    print("\n🏆 OUTSTANDING PERFORMANCE!")
elif success_rate >= 0.7:
    print("\n✅ EXCELLENT PERFORMANCE!")
elif success_rate >= 0.5:
    print("\n👍 GOOD PERFORMANCE!")
elif success_rate >= 0.3:
    print("\n⚠️  MODERATE PERFORMANCE - Consider more training")
else:
    print("\n❌ POOR PERFORMANCE - Needs significant improvement")

# Quick behavior check - first few actions
print(f"\n🔬 BEHAVIOR CHECK (First episode, first 10 actions):")
obs, _ = env.reset()
print(f"Initial: Agent at {env.agent_pos}, Objects at {list(env.objects.keys())}")

actions = []
action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Pickup", 5: "Drop"}

for step in range(10):
    action, _ = model.predict(obs, deterministic=False)
    actions.append(int(action))
    obs, reward, terminated, truncated, _ = env.step(action)
    
    action_name = action_names.get(int(action), "?")
    
    print(f"Step {step+1}: {action_name} → Agent: {env.agent_pos}, Carrying: {env.carried_object}")
    
    if terminated or truncated:
        print(f"Episode ended early at step {step+1}")
        break

print(f"\n🎯 Agent behavior looks {'GOOD' if len(set(actions)) > 2 else 'REPETITIVE'}")
print("✅ Evaluation completed!")