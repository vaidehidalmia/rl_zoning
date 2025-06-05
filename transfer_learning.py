"""
Transfer Learning Curriculum - FIXED VERSION with proper directory handling
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
    smart_transfer_weights,
    load_model_safe,
    create_stage_directories,  # FIXED: Use proper directory creation
)


def create_curriculum_env(grid_size, num_objects):
    """Create environment for curriculum stage"""

    def make_curriculum_env():
        env = make_env(grid_size=grid_size, num_objects=num_objects)
        return env

    return make_curriculum_env


def train_stage_with_transfer(stage, grid_size, num_objects, previous_model_path=None):
    """Train curriculum stage with transfer learning - FIXED directory handling"""
    print(
        f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid, {num_objects} Objects"
    )
    print("=" * 60)

    # FIXED: Create all directories properly FIRST
    stage_dir, log_dir = create_stage_directories(grid_size, num_objects, stage)

    # Calculate parameters
    timesteps, max_steps, learning_rate, net_arch, ent_coef = calculate_training_params(
        grid_size, num_objects
    )
    ppo_params = get_fixed_ppo_params(grid_size, num_objects)

    print(f"📊 Training metrics:")
    print(f"   Complexity: {grid_size * grid_size * num_objects}")
    print(f"   Training steps: {timesteps:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Exploration: {ent_coef}")

    # Determine number of environments
    complexity = grid_size * grid_size * num_objects
    if complexity <= 25:
        n_envs = 1
        vec_env_cls = None
    else:
        n_envs = min(4, max(2, 16 // grid_size))
        vec_env_cls = SubprocVecEnv

    # Create environment
    make_env_func = create_curriculum_env(grid_size, num_objects)
    env = make_vec_env(make_env_func, n_envs=n_envs, vec_env_cls=vec_env_cls)

    print(f"🔧 Environment setup:")
    print(f"   Using {n_envs} parallel environments")

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        **ppo_params,
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    # Transfer learning
    transfer_successful = False
    if previous_model_path and os.path.exists(previous_model_path + ".zip"):
        print(f"🔄 TRANSFER LEARNING: Loading from {previous_model_path}")
        try:
            source_model = PPO.load(previous_model_path, device="cpu")
            transfer_successful = smart_transfer_weights(source_model, model)

            if transfer_successful:
                # Fine-tune with lower learning rate
                model.learning_rate = learning_rate * 0.5
                print(
                    f"🔧 Reduced learning rate to {model.learning_rate} for fine-tuning"
                )

        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")

    if not transfer_successful:
        print("🆕 Training from scratch")

    # FIXED: Setup evaluation with directories that now exist
    eval_env = make_vec_env(make_env_func, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{stage_dir}/best",  # Directory already created
        log_path=log_dir,  # Directory already created
        eval_freq=max(5000, timesteps // 20),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Train
    transfer_note = "with transfer learning" if transfer_successful else "from scratch"
    print(f"🚀 Training {transfer_note} for {timesteps:,} timesteps...")

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


def evaluate_stage(model_path, grid_size, num_objects, episodes=15):
    """Evaluate a trained stage"""
    model = load_model_safe(model_path)
    if model is None:
        print(f"   ❌ Could not load {model_path}")
        return 0.0, 0.0, 0.0

    print(f"   📈 Loaded model from {model_path}")

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env = make_env(grid_size=grid_size, num_objects=num_objects)
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                if terminated:
                    successes += 1
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(env.max_steps)

        total_rewards.append(episode_reward)

    success_rate = successes / episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print(
        f"   📊 Results: {success_rate:.1%} success, {avg_reward:.1f} avg reward, {avg_length:.1f} avg steps"
    )

    return success_rate, avg_reward, avg_length


def create_transfer_curriculum():
    """Create curriculum optimized for transfer learning"""
    curriculum = []

    for grid_size in CURRICULUM_GRID_SIZES:
        max_objects = CURRICULUM_MAX_OBJECTS[grid_size]

        # Progressive object count for each grid size
        for num_objects in range(1, max_objects + 1):
            curriculum.append((grid_size, num_objects))

    return curriculum


def train_transfer_curriculum():
    """Train complete curriculum with transfer learning - FIXED"""
    # FIXED: Create base directories at the start
    from utils import create_directories

    create_directories()

    curriculum = create_transfer_curriculum()

    print(f"🎓 TRANSFER LEARNING CURRICULUM ({len(curriculum)} stages)")
    print("=" * 60)
    for i, (grid_size, num_objects) in enumerate(curriculum, 1):
        print(f"Stage {i:2d}: {grid_size}x{grid_size} grid, {num_objects} objects")
    print("=" * 60)

    results = {}
    successful_stages = 0
    previous_model_path = None

    for stage, (grid_size, num_objects) in enumerate(curriculum, 1):
        stage_key = f"{grid_size}x{grid_size}_{num_objects}obj"

        print(f"\n{'=' * 70}")
        print(f"STARTING STAGE {stage}/{len(curriculum)}: {stage_key}")
        if previous_model_path:
            print(f"Building on: {previous_model_path}")
        print(f"{'=' * 70}")

        # Train stage with transfer learning
        model_path = train_stage_with_transfer(
            stage, grid_size, num_objects, previous_model_path
        )

        if model_path is None:
            print(f"❌ Stage {stage} failed - skipping evaluation")
            results[stage_key] = (0.0, 0.0, 0.0)
            previous_model_path = None  # Don't transfer from failed stage
            continue

        # Evaluate stage
        print(f"\n📊 Evaluating {stage_key} performance...")
        success_rate, avg_reward, avg_length = evaluate_stage(
            model_path, grid_size, num_objects, episodes=EVAL_EPISODES
        )
        results[stage_key] = (success_rate, avg_reward, avg_length)

        # Determine transfer strategy for next stage
        if success_rate >= 0.8:
            print(f"🏆 Stage {stage} EXCELLENT! ({success_rate:.1%} success)")
            successful_stages += 1
            previous_model_path = model_path  # Transfer this excellent model
        elif success_rate >= 0.6:
            print(f"✅ Stage {stage} GOOD! ({success_rate:.1%} success)")
            successful_stages += 1
            previous_model_path = model_path  # Transfer this good model
        elif success_rate >= 0.4:
            print(f"⚠️  Stage {stage} MODERATE ({success_rate:.1%} success)")
            previous_model_path = None  # Don't transfer moderate performance
        else:
            print(f"❌ Stage {stage} POOR ({success_rate:.1%} success)")
            print("   Consider: more training time or curriculum adjustment")
            previous_model_path = None  # Don't transfer poor performance

            # Early stopping for very poor performance
            if success_rate < 0.1 and stage > 3:
                print(
                    f"🛑 Very poor performance detected. Consider revising curriculum."
                )
                break

    # Final comprehensive summary
    print(f"\n🏆 TRANSFER CURRICULUM COMPLETION SUMMARY")
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
        print(f"\n⚠️  No configuration achieved 60%+ success.")

    return results


def quick_test_transfer(grid_size=None, num_objects=None):
    """Quick test with transfer learning - FIXED"""
    grid_size = grid_size or GRID_SIZE
    num_objects = num_objects or NUM_OBJECTS

    print(
        f"🧪 Quick transfer test: {grid_size}x{grid_size} grid with {num_objects} objects"
    )

    # FIXED: Create base directories first
    from utils import create_directories

    create_directories()

    # If multi-object, first train single-object for transfer
    if num_objects > 1:
        print(
            f"🔄 Training base model ({grid_size}x{grid_size}, 1 object) for transfer..."
        )
        base_model_path = train_stage_with_transfer(1, grid_size, 1, None)

        if base_model_path:
            base_success, _, _ = evaluate_stage(base_model_path, grid_size, 1, 10)
            print(f"   Base model success: {base_success:.1%}")

            if base_success >= 0.8:
                print(f"✅ Base model ready for transfer!")
                target_model_path = train_stage_with_transfer(
                    2, grid_size, num_objects, base_model_path
                )
            else:
                print(f"⚠️  Base model not good enough, training target from scratch")
                target_model_path = train_stage_with_transfer(
                    2, grid_size, num_objects, None
                )
        else:
            print(f"❌ Base model training failed")
            return
    else:
        target_model_path = train_stage_with_transfer(1, grid_size, num_objects, None)

    if target_model_path:
        success_rate, avg_reward, avg_length = evaluate_stage(
            target_model_path, grid_size, num_objects, 20
        )
        print(f"\n🏆 Final result: {success_rate:.1%} success rate")
        return success_rate
    else:
        print(f"❌ Training failed")
        return 0.0


def save_curriculum_results(results, filename="transfer_curriculum_results.json"):
    """Save curriculum results to file"""
    serializable_results = {k: list(v) for k, v in results.items()}

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"📁 Results saved to {filename}")


if __name__ == "__main__":
    print("🔄 TRANSFER LEARNING CURRICULUM SYSTEM (FIXED)")
    print("Proper directory handling - should work without errors!")
    print("=" * 60)

    # Print current config
    from config import print_config

    print_config()

    # Choose training mode
    mode = input(
        "Choose mode: [1] Full transfer curriculum, [2] Quick test, [3] Custom config: "
    ).strip()

    if mode == "2":
        quick_test_transfer()

    elif mode == "3":
        try:
            grid_size = int(
                input(f"Grid size (4-8, default {GRID_SIZE}): ") or GRID_SIZE
            )
            num_objects = int(
                input(f"Number of objects (1-3, default {NUM_OBJECTS}): ")
                or NUM_OBJECTS
            )
            quick_test_transfer(grid_size, num_objects)
        except ValueError:
            print("Invalid input, running quick test with config defaults instead")
            quick_test_transfer()

    else:
        # Full curriculum with transfer learning
        results = train_transfer_curriculum()

        # Save results
        save_results = input("\nSave results to file? (y/n): ").strip().lower()
        if save_results == "y":
            save_curriculum_results(results)

        print(f"\n🎉 TRANSFER LEARNING CURRICULUM COMPLETE!")
        print(f"Should now work without directory errors!")
