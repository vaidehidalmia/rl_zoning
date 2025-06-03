"""
Enhanced Fixed Grid Size Curriculum: Progressive scaling for larger grids and more objects

Key improvements:
- Fixed function call bugs
- Dynamic hyperparameter scaling
- Better curriculum progression
- Adaptive training times based on complexity
- Robust evaluation system
- Support for much larger grids (up to 12x12+)
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from environment import ZoningEnv

def calculate_training_params(grid_size, num_objects):
    """Dynamically calculate training parameters based on grid size and object count"""
    # Base complexity factor
    complexity = grid_size * grid_size * num_objects
    
    # Scale timesteps with complexity
    base_timesteps = 150_000
    timesteps = int(base_timesteps * (1 + complexity / 50))
    timesteps = min(timesteps, 2_000_000)  # Cap at 2M
    
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
    
    # Adaptive exploration
    if complexity < 30:
        ent_coef = 0.1
    elif complexity < 80:
        ent_coef = 0.05
    else:
        ent_coef = 0.02
    
    return timesteps, max_steps, learning_rate, net_arch, ent_coef

def train_stage(stage, grid_size, num_objects):
    """Train one curriculum stage - creates fresh model for each grid size"""
    print(f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid, {num_objects} Objects")
    print("=" * 60)
    
    # Calculate adaptive parameters
    timesteps, max_steps, learning_rate, net_arch, ent_coef = calculate_training_params(grid_size, num_objects)
    
    print(f"📊 Complexity metrics:")
    print(f"   Grid: {grid_size}x{grid_size} = {grid_size*grid_size} cells")
    print(f"   Objects: {num_objects}")
    print(f"   Training steps: {timesteps:,}")
    print(f"   Episode max steps: {max_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Exploration: {ent_coef}")
    
    # Create environment with current grid size
    def make_env():
        env = ZoningEnv(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = max_steps
        return env
    
    # Use more environments for larger grids
    n_envs = min(8, max(2, 16 // grid_size))  # Scale down envs for larger grids
    env = make_vec_env(make_env, n_envs=n_envs)
    
    print(f"🆕 Creating new model for {grid_size}x{grid_size} with {num_objects} objects")
    print(f"   Using {n_envs} parallel environments")
    
    # Create model with adaptive parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=0.99,
        clip_range=0.2,
        batch_size=min(512, max(64, 2048 // grid_size)),  # Adaptive batch size
        n_steps=max(512, 4096 // grid_size),  # Adaptive rollout length
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
        device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
    )
    
    # Setup evaluation
    eval_env = make_vec_env(make_env, n_envs=1)
    
    # Create stage-specific directories
    stage_dir = f"models/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}"
    log_dir = f"logs/stage_{stage}_grid_{grid_size}x{grid_size}_obj_{num_objects}/"
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{stage_dir}/best",
        log_path=log_dir,
        eval_freq=max(5000, timesteps // 20),  # Evaluate 20 times during training
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    print(f"🚀 Training {grid_size}x{grid_size} with {num_objects} objects for {timesteps:,} timesteps...")
    
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
    try:
        model = PPO.load(model_path)
        print(f"   📈 Loaded model from {model_path}")
    except Exception as e:
        print(f"   ❌ Could not load {model_path}: {e}")
        return 0.0
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    _, max_steps, _, _, _ = calculate_training_params(grid_size, num_objects)
    
    for episode in range(episodes):
        env = ZoningEnv(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = max_steps
        
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
    
    print(f"   📊 Results: {success_rate:.1%} success, {avg_reward:.1f} avg reward, {avg_length:.1f} avg steps")
    
    return success_rate, avg_reward, avg_length

def create_progressive_curriculum():
    """Create a progressive curriculum for grid sizes and object counts"""
    curriculum = []
    
    # Start with small grids, few objects
    grid_sizes = [4, 5, 6, 7, 8, 9, 10, 12]
    max_objects_per_grid = {4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 12: 10}
    
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
    """Train agent across increasing grid sizes and object counts"""
    
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
        
        print(f"\n{'='*70}")
        print(f"STARTING STAGE {stage}/{len(curriculum)}: {stage_key}")
        print(f"{'='*70}")
        
        # Train stage
        model_path = train_stage(stage, grid_size, num_objects)
        
        if model_path is None:
            print(f"❌ Stage {stage} failed - skipping evaluation")
            results[stage_key] = (0.0, 0.0, 0.0)
            continue
        
        # Evaluate stage
        print(f"\n📊 Evaluating {stage_key} performance...")
        success_rate, avg_reward, avg_length = evaluate_grid_size(
            model_path, grid_size, num_objects, episodes=20
        )
        results[stage_key] = (success_rate, avg_reward, avg_length)
        
        # Assess performance
        if success_rate >= 0.8:
            print(f"🏆 Stage {stage} EXCELLENT! ({success_rate:.1%} success)")
            successful_stages += 1
        elif success_rate >= 0.6:
            print(f"✅ Stage {stage} GOOD! ({success_rate:.1%} success)")
            successful_stages += 1
        elif success_rate >= 0.4:
            print(f"⚠️  Stage {stage} MODERATE ({success_rate:.1%} success)")
        else:
            print(f"❌ Stage {stage} POOR ({success_rate:.1%} success)")
            print("   Consider: more training time or curriculum adjustment")
            
            # Early stopping for very poor performance
            if success_rate < 0.1 and stage > 3:
                print(f"🛑 Very poor performance detected. Consider revising curriculum.")
                break
    
    # Final comprehensive summary
    print(f"\n🏆 CURRICULUM COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Completed: {successful_stages}/{len(curriculum)} stages successfully")
    print(f"Success threshold: 60%+ success rate")
    print()
    
    for stage_key, (success_rate, avg_reward, avg_length) in results.items():
        status = "🏆" if success_rate >= 0.8 else "✅" if success_rate >= 0.6 else "⚠️" if success_rate >= 0.4 else "❌"
        print(f"{status} {stage_key:15s}: {success_rate:5.1%} success | {avg_reward:6.1f} reward | {avg_length:5.1f} steps")
    
    # Find the most complex successfully mastered stage
    successful_configs = [(key, rate) for key, (rate, _, _) in results.items() if rate >= 0.6]
    if successful_configs:
        # Extract grid sizes and object counts
        best_complexity = 0
        best_config = None
        
        for config, rate in successful_configs:
            grid_part, obj_part = config.split('_')
            grid_size = int(grid_part.split('x')[0])
            num_objects = int(obj_part.replace('obj', ''))
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

def quick_test_specific_config(grid_size=6, num_objects=2, timesteps=100_000):
    """Quick test of a specific configuration"""
    print(f"🧪 Quick test: {grid_size}x{grid_size} grid with {num_objects} objects")
    
    timesteps_calc, max_steps, learning_rate, net_arch, ent_coef = calculate_training_params(grid_size, num_objects)
    
    # Use provided timesteps or calculated
    timesteps = min(timesteps, timesteps_calc)
    
    def make_env():
        env = ZoningEnv(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = max_steps
        return env
    
    env = make_vec_env(make_env, n_envs=4)
    
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=128,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch)
    )
    
    print(f"Training {grid_size}x{grid_size} with {num_objects} objects for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps)
    
    # Quick save for evaluation
    temp_path = "temp_test_model"
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

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True) 
    os.makedirs("logs", exist_ok=True)
    
    print("🎓 ENHANCED Grid Size and Object Count Curriculum Learning")
    print("Progressive scaling with adaptive hyperparameters")
    print("Supports larger grids (up to 12x12+) and multiple objects")
    print()
    
    # Choose training mode
    mode = input("Choose mode: [1] Full curriculum, [2] Quick test, [3] Custom config: ").strip()
    
    if mode == "2":
        # Quick test mode
        quick_test_specific_config()
    elif mode == "3":
        # Custom configuration
        try:
            grid_size = int(input("Grid size (4-12): "))
            num_objects = int(input("Number of objects (1-10): "))
            timesteps = int(input("Training timesteps (50000-500000): "))
            quick_test_specific_config(grid_size, num_objects, timesteps)
        except ValueError:
            print("Invalid input, running quick test instead")
            quick_test_specific_config()
    else:
        # Full curriculum (default)
        results = train_grid_curriculum()
        
        # Optionally save results
        save_results = input("\nSave results to file? (y/n): ").strip().lower()
        if save_results == 'y':
            import json
            with open("curriculum_results.json", "w") as f:
                # Convert results to serializable format
                serializable_results = {k: list(v) for k, v in results.items()}
                json.dump(serializable_results, f, indent=2)
            print("Results saved to curriculum_results.json")