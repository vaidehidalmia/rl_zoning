# Final hyperparameter tuning to get from 66.7% to 90%+

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from config import *
from utils import make_env, create_directories
import os


def get_optimized_hyperparameters():
    """Final optimized hyperparameters based on the 66.7% baseline"""
    return {
        "learning_rate": 3e-4,
        "n_steps": 4096,  # Even longer rollouts for better learning
        "batch_size": 64,  # Keep small batch size
        "n_epochs": 20,  # More gradient updates per batch
        "gamma": 0.995,  # Slightly higher discount for longer episodes
        "gae_lambda": 0.98,  # Higher GAE lambda for better advantage estimation
        "clip_range": 0.1,  # Smaller clip range for more conservative updates
        "ent_coef": 0.001,  # Even less exploration - more exploitation
        "vf_coef": 1.0,  # Higher value function coefficient
        "max_grad_norm": 0.5,
        "net_arch": [64, 64],
    }


def train_optimized():
    """Train with final optimized hyperparameters"""
    print("🎯 FINAL OPTIMIZATION: 4x4 Grid, 1 Object")
    print("=" * 60)

    params = get_optimized_hyperparameters()
    timesteps = 150_000  # A bit more training time

    print("📊 Optimized hyperparameters:")
    for key, value in params.items():
        if key != "net_arch":
            print(f"   {key}: {value}")
    print(f"   net_arch: {params['net_arch']}")
    print(f"   timesteps: {timesteps:,}")

    # Create environment
    def make_curriculum_env():
        return make_env(grid_size=4, num_objects=1)

    env = make_vec_env(make_curriculum_env, n_envs=1)

    # Extract net_arch for policy_kwargs
    net_arch = params.pop("net_arch")

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **params,
        policy_kwargs=dict(
            net_arch=net_arch,
            activation_fn="tanh",  # Try tanh activation
        ),
        device="cpu",
    )

    # Setup evaluation during training
    eval_env = make_vec_env(make_curriculum_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1,
    )

    print("🚀 Training with optimized hyperparameters...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    # Final evaluation
    print("\n📊 Final comprehensive evaluation...")
    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(50):  # More episodes for confidence
        test_env = make_env(grid_size=4, num_objects=1)
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

    success_rate = successes / 50
    avg_reward = sum(total_rewards) / 50
    avg_length = sum(episode_lengths) / 50

    print(f"\n🏆 FINAL OPTIMIZED RESULTS:")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:.1f}")
    print(f"Average steps: {avg_length:.1f}")

    env.close()
    eval_env.close()

    return success_rate, model


def test_different_approaches():
    """Test several different approaches to find the best"""
    print("🧪 TESTING MULTIPLE APPROACHES")
    print("=" * 60)

    approaches = [
        {
            "name": "Longer Training",
            "params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "ent_coef": 0.01,
                "net_arch": [64, 64],
            },
            "timesteps": 200_000,
        },
        {
            "name": "Lower Learning Rate",
            "params": {
                "learning_rate": 1e-4,  # Lower LR
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "ent_coef": 0.01,
                "net_arch": [64, 64],
            },
            "timesteps": 150_000,
        },
        {
            "name": "More Gradient Steps",
            "params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 32,  # Even smaller batch
                "n_epochs": 20,  # More epochs
                "ent_coef": 0.001,  # Less exploration
                "net_arch": [64, 64],
            },
            "timesteps": 150_000,
        },
    ]

    results = {}

    for approach in approaches:
        print(f"\n🔬 Testing: {approach['name']}")
        print("-" * 40)

        # Create environment
        def make_curriculum_env():
            return make_env(grid_size=4, num_objects=1)

        env = make_vec_env(make_curriculum_env, n_envs=1)

        # Extract net_arch
        params = approach["params"].copy()
        net_arch = params.pop("net_arch")

        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,  # Less verbose for comparison
            **params,
            policy_kwargs=dict(net_arch=net_arch),
            device="cpu",
        )

        # Train
        model.learn(total_timesteps=approach["timesteps"])

        # Quick evaluation
        successes = 0
        for episode in range(20):
            test_env = make_env(grid_size=4, num_objects=1)
            obs, _ = test_env.reset()

            for step in range(test_env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)

                if terminated:
                    successes += 1
                    break

        success_rate = successes / 20
        results[approach["name"]] = success_rate

        print(f"   Success rate: {success_rate:.1%}")

        env.close()

    # Find best approach
    best_approach = max(results, key=results.get)
    best_rate = results[best_approach]

    print(f"\n🏆 BEST APPROACH: {best_approach}")
    print(f"Success rate: {best_rate:.1%}")

    return results


def analyze_failure_cases():
    """Analyze why the agent fails in some episodes"""
    print("\n🔍 ANALYZING FAILURE CASES")
    print("=" * 40)

    # Train a quick model
    def make_curriculum_env():
        return make_env(grid_size=4, num_objects=1)

    env = make_vec_env(make_curriculum_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
    )

    model.learn(total_timesteps=100_000)
    env.close()

    # Analyze episodes
    successes = 0
    failures = []

    for episode in range(20):
        test_env = make_env(grid_size=4, num_objects=1)
        obs, _ = test_env.reset()

        episode_actions = []
        episode_positions = []

        for step in range(test_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action)
            episode_positions.append(test_env.agent_pos)

            obs, reward, terminated, truncated, _ = test_env.step(action)

            if terminated:
                successes += 1
                break
            elif truncated:
                failures.append(
                    {
                        "episode": episode,
                        "actions": episode_actions,
                        "positions": episode_positions,
                        "final_state": {
                            "agent_pos": test_env.agent_pos,
                            "carried": test_env.carried_object,
                            "objects": test_env.objects.copy(),
                        },
                    }
                )
                break

    print(f"Successes: {successes}/20 = {successes / 20:.1%}")
    print(f"Failures: {len(failures)}")

    if failures:
        print(f"\nFailure analysis (first failure):")
        failure = failures[0]
        print(f"  Final agent position: {failure['final_state']['agent_pos']}")
        print(f"  Carrying object: {failure['final_state']['carried']}")
        print(f"  Objects remaining: {failure['final_state']['objects']}")
        print(f"  Last 10 actions: {failure['actions'][-10:]}")

        # Check if agent got stuck
        recent_positions = (
            failure["positions"][-20:]
            if len(failure["positions"]) >= 20
            else failure["positions"]
        )
        unique_positions = len(set(recent_positions))
        if unique_positions < 5:
            print(
                f"  🚨 Agent appears stuck! Only {unique_positions} unique positions in last moves"
            )


if __name__ == "__main__":
    create_directories()

    print("🎯 FINAL HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Choice of analysis
    choice = input(
        "Choose: [1] Quick optimization, [2] Test multiple approaches, [3] Analyze failures: "
    ).strip()

    if choice == "2":
        results = test_different_approaches()
        print(f"\n📊 All results: {results}")
    elif choice == "3":
        analyze_failure_cases()
    else:
        # Default: quick optimization
        success_rate, model = train_optimized()

        if success_rate >= 0.9:
            print("🎉 EXCELLENT! 90%+ success achieved!")
        elif success_rate >= 0.8:
            print("✅ VERY GOOD! 80%+ success achieved!")
        elif success_rate > 0.667:
            print("✅ IMPROVED! Better than baseline!")
        else:
            print("🤔 Need more tuning...")

        # Save the successful hyperparameters
        if success_rate >= 0.8:
            print(f"\n💾 These hyperparameters work well for simple problems:")
            params = get_optimized_hyperparameters()
            for key, value in params.items():
                print(f"   {key}: {value}")
