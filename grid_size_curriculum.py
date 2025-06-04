"""
Enhanced Grid Size Curriculum - Now using centralized config and utils!

Key improvements:
- Uses config.py for all settings
- Uses utils.py for common functions
- Cleaner, more maintainable code
- Consistent with other scripts
"""

import numpy as np
import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import *
from utils import (
    make_env,
    get_model_path,
    get_log_path,
    calculate_training_params,
    get_fixed_ppo_params,
    load_model_safe,
    create_directories,
)


def create_curriculum_env(grid_size, num_objects):
    """Create environment for curriculum stage using utils"""

    def make_curriculum_env():
        env = make_env(grid_size=grid_size, num_objects=num_objects)
        # Calculate max steps for this specific grid size and object count
        env.max_steps = grid_size * grid_size * 4 + num_objects * 50
        return env

    return make_curriculum_env


def train_stage(stage, grid_size, num_objects):
    """Train one curriculum stage - creates fresh model for each grid size"""
    print(
        f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid, {num_objects} Objects"
    )
    print("=" * 60)

    # Calculate adaptive parameters using utils
    timesteps, max_steps, learning_rate, net_arch, ent_coef = calculate_training_params(
        grid_size, num_objects
    )

    ppo_params = get_fixed_ppo_params(grid_size, num_objects)

    print("📊 Complexity metrics:")
    print(f"   Grid: {grid_size}x{grid_size} = {grid_size * grid_size} cells")
    print(f"   Objects: {num_objects}")
    print(f"   Training steps: {timesteps:,}")
    print(f"   Episode max steps: {max_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Exploration: {ent_coef}")

    # Create environment factory for this stage
    make_env_func = create_curriculum_env(grid_size, num_objects)

    # FIXED: Use single environment for simple problems
    complexity = grid_size * grid_size * num_objects
    if complexity <= 16:
        # Single environment works better for simple problems
        n_envs = 1
        env = make_vec_env(make_env_func, n_envs=n_envs)
    else:
        # Multiple environments for complex problems
        n_envs = min(8, max(2, 16 // grid_size))
        env = make_vec_env(
            make_env_func,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )

    print(
        f"🆕 Creating new model for {grid_size}x{grid_size} with {num_objects} objects"
    )
    print(f"   Using {n_envs} parallel environments")

    # Create model with adaptive parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        **ppo_params,  # Use the fixed PPO params
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    # Setup paths using utils
    stage_dir = get_model_path(grid_size, num_objects, stage)
    log_dir = get_log_path(grid_size, num_objects, stage)

    # Create directories
    os.makedirs(f"{stage_dir}/best", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup evaluation
    eval_env = make_vec_env(make_env_func, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{stage_dir}/best",
        log_path=log_dir,
        eval_freq=max(5000, timesteps // 20),  # Evaluate 20 times during training
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Train
    print(
        f"🚀 Training {grid_size}x{grid_size} with {num_objects} objects for {timesteps:,} timesteps..."
    )

    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback)

        # Save stage model
        stage_path = f"{stage_dir}/final"
        model.save(stage_path)
        print(f"✅ Saved model: {stage_path}")

        env.close()
        eval_env.close()

        return stage_path

    except Exception as e:
        print(f"❌ Training failed for stage {stage}: {e}")
        env.close()
        eval_env.close()
        return None


def evaluate_grid_size(model_path, grid_size, num_objects, episodes=15):
    """Thorough evaluation of a model on specific grid size and object count"""

    # Load model safely using utils
    model = load_model_safe(model_path)
    if model is None:
        print(f"   ❌ Could not load {model_path}")
        return 0.0, 0.0, 0.0

    print(f"   📈 Loaded model from {model_path}")

    successes = 0
    total_rewards = []
    episode_lengths = []

    # Calculate max steps for this configuration
    _, max_steps, _, _, _ = calculate_training_params(grid_size, num_objects)

    for episode in range(episodes):
        # Create environment for this specific configuration
        make_env_func = create_curriculum_env(grid_size, num_objects)
        env = make_env_func()

        obs, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                if terminated:  # Successfully completed
                    successes += 1
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(max_steps)

        total_rewards.append(episode_reward)

    success_rate = successes / episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print(
        f"   📊 Results: {success_rate:.1%} success, {avg_reward:.1f} avg reward, {avg_length:.1f} avg steps"
    )

    return success_rate, avg_reward, avg_length


def create_progressive_curriculum():
    """Create a progressive curriculum using config settings"""
    curriculum = []

    # Use curriculum settings from config
    for grid_size in CURRICULUM_GRID_SIZES:
        max_objects = CURRICULUM_MAX_OBJECTS[grid_size]

        # Progressive object count for each grid size
        for num_objects in range(1, max_objects + 1):
            # Skip some intermediate steps for larger grids to keep curriculum manageable
            if grid_size >= 8 and num_objects % 2 == 0 and num_objects < max_objects:
                continue

            curriculum.append((grid_size, num_objects))

    return curriculum


def train_grid_curriculum():
    """Train agent across increasing grid sizes and object counts"""

    # Create directories using utils
    create_directories()

    curriculum = create_progressive_curriculum()

    print(f"🎓 PROGRESSIVE CURRICULUM ({len(curriculum)} stages)")
    print("=" * 60)
    for i, (grid_size, num_objects) in enumerate(curriculum, 1):
        print(f"Stage {i:2d}: {grid_size}x{grid_size} grid, {num_objects} objects")
    print("=" * 60)

    results = {}
    successful_stages = 0

    for stage, (grid_size, num_objects) in enumerate(curriculum, 1):
        stage_key = f"{grid_size}x{grid_size}_{num_objects}obj"

        print(f"\n{'=' * 70}")
        print(f"STARTING STAGE {stage}/{len(curriculum)}: {stage_key}")
        print(f"{'=' * 70}")

        # Train stage
        model_path = train_stage(stage, grid_size, num_objects)

        if model_path is None:
            print(f"❌ Stage {stage} failed - skipping evaluation")
            results[stage_key] = (0.0, 0.0, 0.0)
            continue

        # Evaluate stage
        print(f"\n📊 Evaluating {stage_key} performance...")
        success_rate, avg_reward, avg_length = evaluate_grid_size(
            model_path, grid_size, num_objects, episodes=EVAL_EPISODES
        )
        results[stage_key] = (success_rate, avg_reward, avg_length)

        # Assess performance using config threshold
        success_threshold = 0.6  # Could be moved to config if needed

        if success_rate >= 0.8:
            print(f"🏆 Stage {stage} EXCELLENT! ({success_rate:.1%} success)")
            successful_stages += 1
        elif success_rate >= success_threshold:
            print(f"✅ Stage {stage} GOOD! ({success_rate:.1%} success)")
            successful_stages += 1
        elif success_rate >= 0.4:
            print(f"⚠️  Stage {stage} MODERATE ({success_rate:.1%} success)")
        else:
            print(f"❌ Stage {stage} POOR ({success_rate:.1%} success)")
            print("   Consider: more training time or curriculum adjustment")

            # Early stopping for very poor performance
            if success_rate < 0.1 and stage > 3:
                print(
                    f"🛑 Very poor performance detected. Consider revising curriculum."
                )
                break

    # Final comprehensive summary
    print(f"\n🏆 CURRICULUM COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Completed: {successful_stages}/{len(curriculum)} stages successfully")
    print(f"Success threshold: 60%+ success rate")
    print()

    for stage_key, (success_rate, avg_reward, avg_length) in results.items():
        status = (
            "🏆"
            if success_rate >= 0.8
            else "✅"
            if success_rate >= 0.6
            else "⚠️"
            if success_rate >= 0.4
            else "❌"
        )
        print(
            f"{status} {stage_key:15s}: {success_rate:5.1%} success | {avg_reward:6.1f} reward | {avg_length:5.1f} steps"
        )

    # Find the most complex successfully mastered stage
    successful_configs = [
        (key, rate) for key, (rate, _, _) in results.items() if rate >= 0.6
    ]
    if successful_configs:
        # Extract grid sizes and object counts
        best_complexity = 0
        best_config = None

        for config, rate in successful_configs:
            grid_part, obj_part = config.split("_")
            grid_size = int(grid_part.split("x")[0])
            num_objects = int(obj_part.replace("obj", ""))
            complexity = grid_size * grid_size * num_objects

            if complexity > best_complexity:
                best_complexity = complexity
                best_config = config

        print(f"\n🎯 Most complex mastered configuration: {best_config}")
        print(f"   Complexity score: {best_complexity}")
    else:
        print(f"\n⚠️  No configuration achieved 60%+ success. Consider:")
        print(f"   - Longer training times")
        print(f"   - Different hyperparameters")
        print(f"   - Simpler curriculum progression")

    return results


def quick_test_specific_config(grid_size=None, num_objects=None, timesteps=None):
    """Quick test of a specific configuration using config defaults"""

    # Use config defaults if not specified
    grid_size = grid_size or GRID_SIZE
    num_objects = num_objects or NUM_OBJECTS

    print(f"🧪 Quick test: {grid_size}x{grid_size} grid with {num_objects} objects")

    timesteps_calc, max_steps, learning_rate, net_arch, ent_coef = (
        calculate_training_params(grid_size, num_objects)
    )

    # Use provided timesteps or calculated
    if timesteps is None:
        timesteps = min(TIMESTEPS, timesteps_calc)

    # Create environment factory
    make_env_func = create_curriculum_env(grid_size, num_objects)
    env = make_vec_env(make_env_func, n_envs=N_ENVS)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=128,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    print(
        f"Training {grid_size}x{grid_size} with {num_objects} objects for {timesteps:,} steps..."
    )
    model.learn(total_timesteps=timesteps)

    # Quick save for evaluation
    temp_path = f"{MODELS_DIR}/temp_test_model"
    model.save(temp_path)

    # Test
    success_rate, avg_reward, avg_length = evaluate_grid_size(
        temp_path, grid_size, num_objects, episodes=10
    )
    print(f"Quick test result: {success_rate:.1%} success rate")

    # Cleanup
    env.close()
    if os.path.exists(temp_path + ".zip"):
        os.remove(temp_path + ".zip")

    return success_rate


def save_curriculum_results(results, filename="curriculum_results.json"):
    """Save curriculum results to file"""
    # Convert results to serializable format
    serializable_results = {k: list(v) for k, v in results.items()}

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"📁 Results saved to {filename}")


if __name__ == "__main__":
    print("🎓 ENHANCED Grid Size and Object Count Curriculum Learning")
    print("Progressive scaling with adaptive hyperparameters")
    print("Now using centralized config and utils!")
    print()

    # Print current config
    print_config()

    # Choose training mode
    mode = input(
        "Choose mode: [1] Full curriculum, [2] Quick test, [3] Custom config: "
    ).strip()

    if mode == "2":
        # Quick test mode using config defaults
        quick_test_specific_config()

    elif mode == "3":
        # Custom configuration
        try:
            grid_size = int(
                input(f"Grid size (4-12, default {GRID_SIZE}): ") or GRID_SIZE
            )
            num_objects = int(
                input(f"Number of objects (1-10, default {NUM_OBJECTS}): ")
                or NUM_OBJECTS
            )
            timesteps = int(
                input(f"Training timesteps (50000-500000, default {TIMESTEPS}): ")
                or TIMESTEPS
            )
            quick_test_specific_config(grid_size, num_objects, timesteps)
        except ValueError:
            print("Invalid input, running quick test with config defaults instead")
            quick_test_specific_config()

    else:
        # Full curriculum (default)
        results = train_grid_curriculum()

        # Save results
        save_results = input("\nSave results to file? (y/n): ").strip().lower()
        if save_results == "y":
            save_curriculum_results(results)
