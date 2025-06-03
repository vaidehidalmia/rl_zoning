"""
Fixed Grid Size Curriculum: 4x4 → 5x5 → 6x6 → 7x7 → 8x8

Creates separate models for each grid size (no direct transfer due to observation space)
but benefits from curriculum progression and optimized hyperparameters
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from environment import ZoningEnv

def train_stage(stage, grid_size, timesteps, max_steps):
    """Train one curriculum stage - creates fresh model for each grid size"""
    print(f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid")
    print("=" * 60)
    
    # Create environment with current grid size
    def make_env():
        env = ZoningEnv(grid_size=grid_size, num_objects=1)
        env.max_steps = max_steps
        return env
    
    env = make_vec_env(make_env, n_envs=4)
    
    # Always create fresh model (observation space changes with grid size)
    print(f"🆕 Creating new model for {grid_size}x{grid_size}")
    
    # Adjust hyperparameters based on stage
    if stage == 1:
        # First stage: standard settings
        learning_rate = 3e-4
        ent_coef = 0.1
        net_arch = [128, 128]
    else:
        # Later stages: more conservative settings
        learning_rate = 2e-4  # Slightly lower LR
        ent_coef = 0.05      # Less exploration
        net_arch = [256, 256] # Larger network for complex grids
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=0.99,
        clip_range=0.2,
        batch_size=256,
        n_steps=2048,
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
    )
    
    # Setup evaluation
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/grid_{grid_size}x{grid_size}_best",
        log_path=f"logs/grid_{grid_size}/",
        eval_freq=15000,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"🚀 Training {grid_size}x{grid_size} for {timesteps} timesteps...")
    print(f"   Settings: LR={learning_rate}, Exploration={ent_coef}, Network={net_arch}")
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save stage model
    stage_path = f"models/grid_{grid_size}x{grid_size}"
    model.save(stage_path)
    print(f"✅ Saved model: {stage_path}")
    
    return stage_path

def evaluate_grid_size(model_path, grid_size, episodes=10):
    """Quick evaluation of a model on specific grid size"""
    from stable_baselines3 import PPO
    
    try:
        model = PPO.load(model_path)
    except:
        print(f"❌ Could not load {model_path}")
        return 0.0
    
    successes = 0
    
    for episode in range(episodes):
        env = ZoningEnv(grid_size=grid_size, num_objects=1)
        env.max_steps = grid_size * grid_size * 3  # Generous time limit
        
        obs, _ = env.reset()
        
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                if terminated:
                    successes += 1
                break
    
    return successes / episodes

def train_grid_curriculum():
    """Train agent across increasing grid sizes"""
    
    # Curriculum: (grid_size, timesteps, max_steps)
    curriculum = [
        (4, 200_000, 200),   # Master 4x4 thoroughly 
        (5, 250_000, 250),   # Learn 5x5
        (6, 300_000, 300),   # Learn 6x6  
        (7, 350_000, 350),   # Learn 7x7
        (8, 400_000, 400),   # Learn 8x8
        (9, 450_000, 450),   # Learn 9x9
        (10, 500_000, 500),   # Master 10x10
    ]
    
    results = {}
    
    for stage, (grid_size, timesteps, max_steps) in enumerate(curriculum, 1):
        print(f"\n{'='*70}")
        print(f"STARTING STAGE {stage}: {grid_size}x{grid_size} Grid")
        print(f"{'='*70}")
        
        # Train stage
        model_path = train_stage(stage, grid_size, timesteps, max_steps)
        
        # Evaluate stage
        print(f"\n📊 Evaluating {grid_size}x{grid_size} performance...")
        success_rate = evaluate_grid_size(model_path, grid_size, episodes=15)
        results[grid_size] = success_rate
        
        if success_rate >= 0.7:
            print(f"🏆 Stage {stage} EXCELLENT! ({success_rate:.1%} success)")
        elif success_rate >= 0.5:
            print(f"✅ Stage {stage} GOOD! ({success_rate:.1%} success)")
        elif success_rate >= 0.3:
            print(f"⚠️  Stage {stage} MODERATE ({success_rate:.1%} success)")
        else:
            print(f"❌ Stage {stage} POOR ({success_rate:.1%} success)")
            print("   Consider: more training time, different hyperparameters, or easier curriculum")
    
    # Final summary
    print(f"\n🏆 CURRICULUM SUMMARY")
    print("=" * 50)
    
    for grid_size, success_rate in results.items():
        status = "🏆" if success_rate >= 0.8 else "✅" if success_rate >= 0.6 else "⚠️" if success_rate >= 0.4 else "❌"
        print(f"{status} {grid_size}x{grid_size}: {success_rate:.1%} success rate")
    
    # Find best achieved size
    successful_sizes = [size for size, rate in results.items() if rate >= 0.6]
    if successful_sizes:
        max_size = max(successful_sizes)
        print(f"\n🎯 Successfully mastered up to: {max_size}x{max_size} grids!")
    else:
        print(f"\n⚠️  No grid size achieved 60%+ success. Consider adjusting approach.")
    
    return f"models/grid_8x8"

def quick_test_curriculum():
    """Quick test of a specific grid size"""
    grid_size = 6  # Test 6x6
    
    print(f"🧪 Quick test: {grid_size}x{grid_size} grid")
    
    # Train small model
    def make_env():
        env = ZoningEnv(grid_size=grid_size, num_objects=1)
        env.max_steps = grid_size * grid_size * 2
        return env
    
    env = make_vec_env(make_env, n_envs=2)
    
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4, n_steps=1024, batch_size=128,
        ent_coef=0.1, policy_kwargs=dict(net_arch=[128, 128])
    )
    
    print(f"Training on {grid_size}x{grid_size} for 50k steps...")
    model.learn(total_timesteps=50_000)
    
    # Test
    success_rate = evaluate_grid_size("temp_model", grid_size, episodes=5)
    print(f"Result: {success_rate:.1%} success rate")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True) 
    os.makedirs("logs", exist_ok=True)
    
    print("🎓 FIXED Grid Size Curriculum Learning")
    print("Creates separate models for each grid size")
    print("Benefits: Progressive difficulty + optimized hyperparameters")
    
    # Run full curriculum
    train_grid_curriculum()
    
    # Optionally run quick test instead
    # quick_test_curriculum()