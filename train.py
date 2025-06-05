"""
Simple Training Script - Updated to use consistent hyperparameters
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from config import *
from utils import (
    make_env,
    get_fixed_ppo_params,
    calculate_training_params,
    create_directories,
)
import os


def train_simple_model(grid_size=None, num_objects=None, timesteps=None):
    """Train a simple model with optimized hyperparameters"""

    # Use config defaults or provided values
    grid_size = grid_size or GRID_SIZE
    num_objects = num_objects or NUM_OBJECTS

    print(f"🚀 SIMPLE TRAINING: {grid_size}x{grid_size} grid, {num_objects} objects")
    print("=" * 60)

    # Get optimized parameters
    calc_timesteps, max_steps, learning_rate, net_arch, ent_coef = (
        calculate_training_params(grid_size, num_objects)
    )
    ppo_params = get_fixed_ppo_params(grid_size, num_objects)

    # Use provided timesteps or calculated
    if timesteps is None:
        timesteps = calc_timesteps

    print(f"📊 Training parameters:")
    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Objects: {num_objects}")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Entropy coef: {ent_coef}")
    print(f"   Max steps per episode: {max_steps}")

    # Create directories
    create_directories()

    # Environment setup
    def make_simple_env():
        return make_env(grid_size=grid_size, num_objects=num_objects)

    # Use single environment for simple problems, multiple for complex
    complexity = grid_size * grid_size * num_objects
    n_envs = 1 if complexity <= 25 else N_ENVS

    env = make_vec_env(make_simple_env, n_envs=n_envs)

    print(f"   Using {n_envs} parallel environments")

    # Create model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        **ppo_params,
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    # Setup evaluation
    eval_env = make_vec_env(make_simple_env, n_envs=1)

    # Model save path
    model_name = f"simple_{grid_size}x{grid_size}_{num_objects}obj"
    model_path = f"{MODELS_DIR}/{model_name}"
    log_path = f"{LOGS_DIR}/{model_name}/"

    os.makedirs(f"{model_path}_best", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_path}_best",
        log_path=log_path,
        eval_freq=max(5000, timesteps // 20),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    # Train
    print(f"🏃 Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    # Save final model
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")

    # Quick evaluation
    print(f"\n📊 Quick evaluation...")
    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(20):
        test_env = make_env(grid_size=grid_size, num_objects=num_objects)
        obs, _ = test_env.reset()
        episode_reward = 0

        for step in range(test_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                if terminated:
                    successes += 1
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(test_env.max_steps)

        total_rewards.append(episode_reward)

    success_rate = successes / 20
    avg_reward = sum(total_rewards) / 20
    avg_length = sum(episode_lengths) / 20

    print(f"🏆 TRAINING RESULTS:")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Average reward: {avg_reward:.1f}")
    print(f"   Average episode length: {avg_length:.1f} steps")

    env.close()
    eval_env.close()

    if success_rate >= 0.9:
        print("🎉 EXCELLENT performance!")
    elif success_rate >= 0.7:
        print("✅ GOOD performance!")
    elif success_rate >= 0.5:
        print("⚠️  MODERATE performance - consider more training")
    else:
        print("❌ POOR performance - check hyperparameters")

    return model_path, success_rate


def compare_configurations():
    """Compare performance across different configurations"""
    print("🔬 COMPARING CONFIGURATIONS")
    print("=" * 60)

    configs = [
        (4, 1),  # Simple
        (4, 2),  # Medium
        (5, 1),  # Larger grid
        (5, 2),  # Complex
    ]

    results = {}

    for grid_size, num_objects in configs:
        print(f"\n📍 Testing {grid_size}x{grid_size} with {num_objects} objects...")

        try:
            model_path, success_rate = train_simple_model(
                grid_size=grid_size,
                num_objects=num_objects,
                timesteps=100_000,  # Reduced for comparison
            )
            results[(grid_size, num_objects)] = success_rate

        except Exception as e:
            print(f"❌ Failed: {e}")
            results[(grid_size, num_objects)] = 0.0

    print(f"\n🏆 CONFIGURATION COMPARISON:")
    print("=" * 40)
    for (grid_size, num_objects), success_rate in results.items():
        complexity = grid_size * grid_size * num_objects
        status = (
            "🏆"
            if success_rate >= 0.9
            else "✅"
            if success_rate >= 0.7
            else "⚠️"
            if success_rate >= 0.5
            else "❌"
        )
        print(
            f"{status} {grid_size}x{grid_size}, {num_objects} obj (complexity {complexity:2d}): {success_rate:.1%}"
        )


if __name__ == "__main__":
    print("🚀 SIMPLE TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 60)

    # Print current config
    print_config()

    # Choose mode
    mode = input(
        "Choose: [1] Simple training, [2] Custom config, [3] Compare configs: "
    ).strip()

    if mode == "2":
        try:
            grid_size = int(
                input(f"Grid size (4-8, default {GRID_SIZE}): ") or GRID_SIZE
            )
            num_objects = int(
                input(f"Objects (1-3, default {NUM_OBJECTS}): ") or NUM_OBJECTS
            )
            timesteps = int(input(f"Timesteps (50k-500k, default auto): ") or 0) or None
            train_simple_model(grid_size, num_objects, timesteps)
        except ValueError:
            print("Invalid input, using defaults")
            train_simple_model()

    elif mode == "3":
        compare_configurations()

    else:
        # Simple training with current config
        train_simple_model()
