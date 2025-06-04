"""
Enhanced Fixed Grid Size Curriculum: Progressive scaling for larger grids and more objects
WITH IMPROVED DETERMINISTIC PERFORMANCE

Key improvements for deterministic performance:
1. Much lower entropy coefficients with decay schedule
2. Significantly increased training time
3. Deterministic evaluation during training
4. Early stopping when deterministic target is reached
"""

from environment import ZoningEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import os


class DeterministicEvalCallback(BaseCallback):
    """Custom callback to monitor deterministic performance during training"""

    def __init__(
        self,
        eval_env,
        grid_size,
        num_objects,
        eval_freq=10000,
        target_success_rate=0.8,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.eval_freq = eval_freq
        self.target_success_rate = target_success_rate
        self.best_det_success = 0.0
        self.evaluations = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Quick deterministic evaluation
            det_success = self._evaluate_deterministic(n_episodes=5)
            self.evaluations.append((self.n_calls, det_success))

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Deterministic success = {det_success:.1%}")

            if det_success > self.best_det_success:
                self.best_det_success = det_success

            # Early stopping if target reached
            if det_success >= self.target_success_rate:
                print(
                    f"🎯 Target deterministic success rate reached: {det_success:.1%}"
                )
                print(f"Stopping training early at step {self.n_calls}")
                return False  # Stop training

        return True

    def _evaluate_deterministic(self, n_episodes=5):
        """Quick deterministic evaluation"""
        successes = 0

        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle new gym API

            for _ in range(self.eval_env.get_attr("max_steps")[0]):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                if terminated[0]:  # Successfully completed
                    successes += 1
                    break

                if truncated[0]:  # Time limit reached
                    break

        return successes / n_episodes


class EntropyDecayCallback(BaseCallback):
    """Callback to decay entropy coefficient during training"""

    def __init__(self, initial_ent, final_ent, decay_steps, verbose=1):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        if self.n_calls <= self.decay_steps:
            # Exponential decay
            progress = self.n_calls / self.decay_steps
            current_ent = (
                self.initial_ent * (self.final_ent / self.initial_ent) ** progress
            )

            # Update the model's entropy coefficient
            self.model.ent_coef = current_ent

            if self.verbose > 1 and self.n_calls % 10000 == 0:
                print(f"Step {self.n_calls}: Entropy coef = {current_ent:.6f}")

        return True


def calculate_training_params(grid_size, num_objects):
    """Dynamically calculate training parameters with focus on deterministic performance"""
    # Base complexity factor
    complexity = grid_size * grid_size * num_objects

    # SIGNIFICANTLY INCREASED training time (Strategy #2)
    base_timesteps = 500_000  # Up from 150_000
    timesteps = int(base_timesteps * (1 + complexity / 30))  # More aggressive scaling
    timesteps = min(timesteps, 5_000_000)  # Higher cap (was 2M)

    # Scale max steps generously
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    # Adaptive learning rate (lower for complex scenarios)
    if complexity < 30:
        learning_rate = 3e-4
    elif complexity < 80:
        learning_rate = 2e-4
    else:
        learning_rate = 1e-4

    # Adaptive network size
    if complexity < 50:
        net_arch = [128, 128]
    elif complexity < 100:
        net_arch = [256, 256]
    else:
        net_arch = [512, 256]

    # MUCH LOWER entropy coefficients (Strategy #1)
    # These will be the starting values for entropy decay
    if complexity < 30:
        initial_ent = 0.02  # Down from 0.1
        final_ent = 0.0005  # Very low final entropy
    elif complexity < 80:
        initial_ent = 0.01  # Down from 0.05
        final_ent = 0.0002  # Very low final entropy
    else:
        initial_ent = 0.005  # Down from 0.02
        final_ent = 0.0001  # Very low final entropy

    return timesteps, max_steps, learning_rate, net_arch, initial_ent, final_ent


def train_stage(stage, grid_size, num_objects):
    """Train one curriculum stage with improved deterministic performance"""
    print(
        f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid, {num_objects} Objects"
    )
    print("=" * 60)

    # Calculate adaptive parameters
    timesteps, max_steps, learning_rate, net_arch, initial_ent, final_ent = (
        calculate_training_params(grid_size, num_objects)
    )

    print(f"📊 Enhanced training metrics:")
    print(f"   Grid: {grid_size}x{grid_size} = {grid_size * grid_size} cells")
    print(f"   Objects: {num_objects}")
    print(f"   Training steps: {timesteps:,} (INCREASED)")
    print(f"   Episode max steps: {max_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Initial entropy: {initial_ent} (REDUCED)")
    print(f"   Final entropy: {final_ent} (VERY LOW)")

    # Create environment with current grid size
    def make_env():
        env = ZoningEnv(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = max_steps
        return env

    # Use more environments for larger grids
    n_envs = min(8, max(2, 16 // grid_size))
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    print(
        f"🆕 Creating new model for {grid_size}x{grid_size} with {num_objects} objects"
    )
    print(f"   Using {n_envs} parallel environments")

    # Create model with adaptive parameters (starting with initial entropy)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=0.99,
        clip_range=0.2,
        batch_size=min(512, max(64, 2048 // grid_size)),
        n_steps=max(512, 4096 // grid_size),
        ent_coef=initial_ent,  # Will be decayed during training
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    # Setup evaluation environment
    eval_env = make_vec_env(make_env, n_envs=1)

    # Create stage-specific directories
    stage_dir = f"models/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}"
    log_dir = f"logs/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}/"
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup callbacks
    callbacks = []

    # 1. Entropy decay callback
    decay_steps = timesteps // 2  # Decay over first half of training
    entropy_callback = EntropyDecayCallback(
        initial_ent=initial_ent, final_ent=final_ent, decay_steps=decay_steps, verbose=1
    )
    callbacks.append(entropy_callback)

    # 2. Deterministic evaluation callback
    det_eval_callback = DeterministicEvalCallback(
        eval_env=eval_env,
        grid_size=grid_size,
        num_objects=num_objects,
        eval_freq=max(5000, timesteps // 30),  # More frequent evaluation
        target_success_rate=0.8,  # Stop when we hit 80% deterministic success
        verbose=1,
    )
    callbacks.append(det_eval_callback)

    # 3. Regular evaluation callback (stochastic)
    regular_eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{stage_dir}/best_stochastic",
        log_path=log_dir,
        eval_freq=max(10000, timesteps // 20),
        deterministic=False,  # Keep stochastic evaluation for comparison
        render=False,
        n_eval_episodes=5,
    )
    callbacks.append(regular_eval_callback)

    # Train with callbacks
    print(
        f"🚀 Training {grid_size}x{grid_size} with {num_objects} objects for up to {timesteps:,} timesteps..."
    )
    print(f"   🎯 Target: 80% deterministic success rate")
    print(f"   📉 Entropy decay: {initial_ent:.4f} → {final_ent:.6f}")

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)

        # Save final model
        final_path = f"{stage_dir}/final"
        model.save(final_path)
        print(f"✅ Saved final model: {final_path}")

        # Save best deterministic model if we have good performance
        if det_eval_callback.best_det_success >= 0.6:
            best_det_path = f"{stage_dir}/best_deterministic"
            model.save(best_det_path)
            print(f"🎯 Saved best deterministic model: {best_det_path}")
            print(
                f"   Best deterministic success: {det_eval_callback.best_det_success:.1%}"
            )

        env.close()
        eval_env.close()

        # Return the best model path for evaluation
        if det_eval_callback.best_det_success >= 0.6:
            return best_det_path
        else:
            return final_path

    except Exception as e:
        print(f"❌ Training failed for stage {stage}: {e}")
        env.close()
        eval_env.close()
        return None


def evaluate_grid_size(
    model_path, grid_size, num_objects, episodes=15, deterministic=True
):
    """Thorough evaluation of a model on specific grid size and object count"""
    try:
        model = PPO.load(model_path)
        print(f"   📈 Loaded model from {model_path}")
    except Exception as e:
        print(f"   ❌ Could not load {model_path}: {e}")
        return 0.0, 0.0, 0.0

    successes = 0
    total_rewards = []
    episode_lengths = []

    timesteps, max_steps, _, _, _, _ = calculate_training_params(grid_size, num_objects)

    eval_mode = "deterministic" if deterministic else "stochastic"
    print(f"   🔍 Evaluating in {eval_mode} mode...")

    for episode in range(episodes):
        env = ZoningEnv(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = max_steps

        obs, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
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
        f"   📊 {eval_mode.title()} results: {success_rate:.1%} success, {avg_reward:.1f} avg reward, {avg_length:.1f} avg steps"
    )

    return success_rate, avg_reward, avg_length


def create_progressive_curriculum():
    """Create a progressive curriculum for grid sizes and object counts"""
    curriculum = []

    # Start with small grids, few objects
    grid_sizes = [4, 5]
    max_objects_per_grid = {4: 3, 5: 4}

    for grid_size in grid_sizes:
        max_objects = max_objects_per_grid[grid_size]

        # Progressive object count for each grid size
        for num_objects in range(1, max_objects + 1):
            # Skip some intermediate steps for larger grids to keep curriculum manageable
            if grid_size >= 8 and num_objects % 2 == 0 and num_objects < max_objects:
                continue

            curriculum.append((grid_size, num_objects))

    return curriculum


def train_grid_curriculum():
    """Train agent across increasing grid sizes and object counts with improved deterministic performance"""

    curriculum = create_progressive_curriculum()

    print(f"🎓 ENHANCED DETERMINISTIC CURRICULUM ({len(curriculum)} stages)")
    print("Progressive scaling with:")
    print("✨ Much lower entropy coefficients with decay")
    print("⏰ Significantly increased training time")
    print("🎯 Deterministic evaluation during training")
    print("🛑 Early stopping when target reached")
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

        # Train stage with enhanced deterministic focus
        model_path = train_stage(stage, grid_size, num_objects)

        if model_path is None:
            print(f"❌ Stage {stage} failed - skipping evaluation")
            results[stage_key] = {"det": (0.0, 0.0, 0.0), "stoch": (0.0, 0.0, 0.0)}
            continue

        # Evaluate stage - BOTH deterministic and stochastic
        print(f"\n📊 Evaluating {stage_key} performance...")

        # Deterministic evaluation (main focus)
        det_success, det_reward, det_length = evaluate_grid_size(
            model_path, grid_size, num_objects, episodes=20, deterministic=True
        )

        # Stochastic evaluation (for comparison)
        stoch_success, stoch_reward, stoch_length = evaluate_grid_size(
            model_path, grid_size, num_objects, episodes=20, deterministic=False
        )

        results[stage_key] = {
            "det": (det_success, det_reward, det_length),
            "stoch": (stoch_success, stoch_reward, stoch_length),
        }

        # Assess performance based on DETERMINISTIC results
        if det_success >= 0.8:
            print(
                f"🏆 Stage {stage} EXCELLENT! ({det_success:.1%} deterministic success)"
            )
            successful_stages += 1
        elif det_success >= 0.6:
            print(f"✅ Stage {stage} GOOD! ({det_success:.1%} deterministic success)")
            successful_stages += 1
        elif det_success >= 0.4:
            print(
                f"⚠️  Stage {stage} MODERATE ({det_success:.1%} deterministic success)"
            )
        else:
            print(f"❌ Stage {stage} POOR ({det_success:.1%} deterministic success)")
            print("   Stochastic performance for comparison:")
            print(f"   Stochastic: {stoch_success:.1%} success")

            # Early stopping for very poor deterministic performance
            if det_success < 0.1 and stage > 3:
                print(
                    f"🛑 Very poor deterministic performance. Consider revising approach."
                )
                break

    # Final comprehensive summary
    print(f"\n🏆 ENHANCED CURRICULUM COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Successfully completed: {successful_stages}/{len(curriculum)} stages")
    print(f"Success threshold: 60%+ DETERMINISTIC success rate")
    print()
    print(f"{'Stage':<15} {'Det Success':<12} {'Stoch Success':<14} {'Status'}")
    print("-" * 70)

    for stage_key, results_dict in results.items():
        det_success, _, _ = results_dict["det"]
        stoch_success, _, _ = results_dict["stoch"]

        status = (
            "🏆 EXCELLENT"
            if det_success >= 0.8
            else "✅ GOOD"
            if det_success >= 0.6
            else "⚠️ MODERATE"
            if det_success >= 0.4
            else "❌ POOR"
        )

        print(
            f"{stage_key:<15} {det_success:>6.1%}      {stoch_success:>6.1%}        {status}"
        )

    return results


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("🎓 ENHANCED Grid Size Curriculum for DETERMINISTIC Performance")
    print("🔥 Key improvements:")
    print("   • Much lower entropy coefficients (0.02 → 0.0001)")
    print("   • Entropy decay during training")
    print("   • 3-5x more training time (500K-5M steps)")
    print("   • Deterministic monitoring during training")
    print("   • Early stopping when deterministic target reached")
    print()

    # Run the enhanced curriculum
    results = train_grid_curriculum()

    # Save results
    save_results = input("\nSave results to file? (y/n): ").strip().lower()
    if save_results == "y":
        import json

        # Convert results to serializable format
        serializable_results = {}
        for stage_key, results_dict in results.items():
            serializable_results[stage_key] = {
                "deterministic": list(results_dict["det"]),
                "stochastic": list(results_dict["stoch"]),
            }

        with open("enhanced_curriculum_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        print("Results saved to enhanced_curriculum_results.json")
