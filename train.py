import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from config import *
from utils import make_env, load_model_safe, create_directories


def train_agent():
    """Train agent using config settings"""

    # Print current configuration
    print_config()

    # Create directories
    create_directories()

    # Create vectorized environment using config
    env = make_vec_env(make_env, n_envs=N_ENVS)

    # Create model using config
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        gamma=0.99,
        n_steps=1024,
        batch_size=256,
        ent_coef=ENT_COEF,
        policy_kwargs=dict(net_arch=NET_ARCH),
        device="cpu",
    )

    # Setup evaluation using config
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best",
        log_path=LOG_PATH,
        eval_freq=max(5000, TIMESTEPS // 20),
        deterministic=True,
        n_eval_episodes=5,
    )

    # Train
    print(f"🚀 Training for {TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)

    # Save final model
    model.save(MODEL_PATH)
    print(f"✅ Saved model: {MODEL_PATH}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def quick_test():
    """Quick test of trained model"""
    model = load_model_safe()
    if model is None:
        print("❌ No trained model found. Run training first!")
        return

    print(f"✅ Loaded model from {MODEL_PATH}")

    env = make_env()

    successes = 0
    for episode in range(10):
        obs, _ = env.reset()
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                successes += 1
                print(f"Episode {episode + 1}: ✅ Completed in {step + 1} steps")
                break
            elif truncated:
                print(f"Episode {episode + 1}: ❌ Timeout")
                break

    success_rate = successes / 10
    print(f"\n📊 Success rate: {success_rate:.1%}")


if __name__ == "__main__":
    print("🤖 Simple Zoning Agent Training")
    print("=" * 40)

    choice = input("Choose: [1] Train, [2] Test, [3] Both: ").strip()

    if choice == "1":
        train_agent()
    elif choice == "2":
        quick_test()
    else:  # Default to both
        train_agent()
        print("\n" + "=" * 40)
        print("🧪 Testing trained model...")
        quick_test()
