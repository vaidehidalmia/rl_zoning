import numpy as np
from config import *
from utils import make_env, load_model_safe


def evaluate_agent(episodes=None):
    """Evaluate trained agent"""

    episodes = episodes or EVAL_EPISODES

    # Try to load model
    model = load_model_safe()
    if model is None:
        print("Make sure you've trained a model first!")
        return

    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"🧪 Evaluating on {GRID_SIZE}x{GRID_SIZE} grid with {NUM_OBJECTS} objects")
    print(f"📊 Running {episodes} episodes...")

    # Track results
    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                if terminated:
                    successes += 1
                    status = "✅ COMPLETED"
                else:
                    status = "❌ TIMEOUT"

                print(
                    f"Episode {episode + 1:2d}: {status} in {step + 1:2d} steps (reward: {episode_reward:6.1f})"
                )
                episode_lengths.append(step + 1)
                break

        total_rewards.append(episode_reward)

    # Calculate statistics
    success_rate = successes / episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"\n📈 RESULTS:")
    print(f"   Success Rate: {success_rate:.1%} ({successes}/{episodes})")
    print(f"   Average Reward: {avg_reward:.1f}")
    print(f"   Average Steps: {avg_length:.1f} ± {std_length:.1f}")
    print(f"   Best Episode: {min(episode_lengths)} steps")
    print(f"   Worst Episode: {max(episode_lengths)} steps")

    # Performance assessment
    if success_rate >= 0.8:
        print("🏆 EXCELLENT performance!")
    elif success_rate >= 0.6:
        print("✅ GOOD performance!")
    elif success_rate >= 0.4:
        print("⚠️  MODERATE performance")
    else:
        print("❌ POOR performance - consider more training")

    return success_rate, avg_reward, avg_length


def watch_agent_play(episodes=3):
    """Watch agent play with visual output"""
    model = load_model_safe()
    if model is None:
        print("❌ No trained model found!")
        return

    for episode in range(episodes):
        print(f"\n🎮 Episode {episode + 1}:")
        env = make_env(render_mode="human")
        obs, _ = env.reset()

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Simple text visualization
            print(f"Step {step + 1}: Action={action}, Reward={reward:.1f}")

            if terminated:
                print(f"✅ Completed in {step + 1} steps!")
                break
            elif truncated:
                print(f"❌ Timeout after {step + 1} steps")
                break


if __name__ == "__main__":
    print("🧪 Agent Evaluation")
    print("=" * 30)
    print_config()

    choice = input("Choose: [1] Evaluate, [2] Watch play, [3] Both: ").strip()

    if choice == "1":
        evaluate_agent()
    elif choice == "2":
        watch_agent_play()
    else:  # Default to both
        evaluate_agent()
        print("\n" + "=" * 30)
        watch_agent_play(episodes=1)
