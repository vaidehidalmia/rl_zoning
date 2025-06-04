# Final fix for the curriculum - proper entropy coefficient


def get_fixed_ppo_params(grid_size, num_objects):
    """FINAL FIXED PPO hyperparameters - higher entropy to prevent loops"""
    complexity = grid_size * grid_size * num_objects

    if complexity <= 16:  # Simple problems
        return {
            "n_steps": 2048,  # Good rollout length
            "batch_size": 64,  # Small batch size
            "n_epochs": 10,  # Standard epochs
            "gamma": 0.99,  # Standard discount
            "gae_lambda": 0.95,  # Standard GAE
            "clip_range": 0.2,  # Standard clip range
            "vf_coef": 0.5,  # Standard value function weight
            "max_grad_norm": 0.5,  # Standard gradient clipping
        }
    else:
        return {
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }


def calculate_training_params(grid_size, num_objects):
    """FINAL FIXED training parameters - proper entropy coefficient"""
    complexity = grid_size * grid_size * num_objects

    if complexity <= 16:  # Simple problems
        base_timesteps = 200_000
        learning_rate = 3e-4
        net_arch = [64, 64]
        ent_coef = 0.05  # INCREASED! Was 0.01, now 0.05 to prevent deterministic loops
    elif complexity <= 50:
        base_timesteps = 300_000
        learning_rate = 3e-4
        net_arch = [128, 128]
        ent_coef = 0.03
    else:
        base_timesteps = 500_000
        learning_rate = 2e-4
        net_arch = [256, 256]
        ent_coef = 0.02

    timesteps = min(base_timesteps, 2_000_000)
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    return timesteps, max_steps, learning_rate, net_arch, ent_coef


# Test the fix
def test_entropy_fix():
    """Test different entropy coefficients to find the sweet spot"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from utils import make_env

    entropy_values = [0.01, 0.03, 0.05, 0.1]

    print("🧪 TESTING ENTROPY COEFFICIENT VALUES")
    print("=" * 50)

    results = {}

    for ent_coef in entropy_values:
        print(f"\n🔬 Testing ent_coef = {ent_coef}")

        # Create environment
        def make_curriculum_env():
            return make_env(grid_size=4, num_objects=1)

        env = make_vec_env(make_curriculum_env, n_envs=1)

        # Create model with this entropy coefficient
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,  # Test this value
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[64, 64]),
            device="cpu",
        )

        # Quick training
        print(f"   Training for 100k steps...")
        model.learn(total_timesteps=100_000)

        # Test deterministic performance
        successes_det = 0
        successes_stoch = 0

        for episode in range(20):
            # Test deterministic
            test_env = make_env(grid_size=4, num_objects=1)
            obs, _ = test_env.reset()

            for step in range(test_env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)

                if terminated:
                    successes_det += 1
                    break

            # Test stochastic
            test_env = make_env(grid_size=4, num_objects=1)
            obs, _ = test_env.reset()

            for step in range(test_env.max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _ = test_env.step(action)

                if terminated:
                    successes_stoch += 1
                    break

        det_rate = successes_det / 20
        stoch_rate = successes_stoch / 20

        print(f"   Deterministic: {det_rate:.1%}")
        print(f"   Stochastic:    {stoch_rate:.1%}")
        print(f"   Difference:    {abs(det_rate - stoch_rate):.1%}")

        results[ent_coef] = (det_rate, stoch_rate)

        env.close()

    # Find best entropy coefficient
    print(f"\n🏆 ENTROPY COEFFICIENT RESULTS:")
    print("=" * 50)

    best_ent_coef = None
    best_det_rate = 0

    for ent_coef, (det_rate, stoch_rate) in results.items():
        difference = abs(det_rate - stoch_rate)
        quality_score = det_rate - difference * 0.5  # Penalize large differences

        print(
            f"ent_coef={ent_coef}: det={det_rate:.1%}, stoch={stoch_rate:.1%}, diff={difference:.1%}, score={quality_score:.3f}"
        )

        if (
            det_rate > best_det_rate and difference < 0.3
        ):  # Good deterministic performance, small difference
            best_det_rate = det_rate
            best_ent_coef = ent_coef

    print(f"\n🎯 RECOMMENDED ent_coef: {best_ent_coef}")
    print(f"   Achieves {best_det_rate:.1%} deterministic success")

    return best_ent_coef


if __name__ == "__main__":
    print("🔧 FINAL CURRICULUM FIX: ENTROPY COEFFICIENT")
    print("=" * 60)

    # Test to find optimal entropy coefficient
    best_ent_coef = test_entropy_fix()

    print(f"\n💾 UPDATE YOUR utils.py:")
    print(f"In calculate_training_params(), change:")
    print(f"   ent_coef = 0.01  # OLD")
    print(f"   ent_coef = {best_ent_coef}  # NEW")
    print(f"\nThis should fix the deterministic loop issue!")
