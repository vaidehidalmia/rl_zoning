# Quick debug script to compare train.py vs curriculum approaches

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from config import *
from utils import make_env, calculate_training_params


def test_train_py_style():
    """Test using train.py style (assuming defaults)"""
    print("🔍 Testing train.py style...")

    # Simple environment creation like train.py probably does
    def make_simple_env():
        return make_env()

    env = make_vec_env(make_simple_env, n_envs=N_ENVS)

    # Default PPO hyperparameters (what train.py likely uses)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,  # From config
        # Using PPO defaults for batch_size, n_steps, etc.
        device="cpu",
    )

    print(f"Default PPO settings:")
    print(f"  batch_size: {model.batch_size}")
    print(f"  n_steps: {model.n_steps}")
    print(f"  learning_rate: {model.learning_rate}")

    # Quick training
    model.learn(total_timesteps=50000)

    # Test
    test_env = make_env()
    success_count = 0

    for episode in range(10):
        obs, _ = test_env.reset()
        for step in range(test_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            if terminated:
                success_count += 1
                break

    success_rate = success_count / 10
    print(f"Train.py style success rate: {success_rate:.1%}")

    env.close()
    return success_rate


def test_curriculum_style():
    """Test using curriculum style"""
    print("\n🔍 Testing curriculum style...")

    grid_size, num_objects = GRID_SIZE, NUM_OBJECTS
    timesteps, max_steps, learning_rate, net_arch, ent_coef = calculate_training_params(
        grid_size, num_objects
    )

    # Curriculum environment creation
    def make_curriculum_env():
        env = make_env(grid_size=grid_size, num_objects=num_objects)
        env.max_steps = grid_size * grid_size * 4 + num_objects * 50
        return env

    n_envs = min(8, max(2, 16 // grid_size))
    env = make_vec_env(make_curriculum_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Curriculum hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=0.99,
        clip_range=0.2,
        batch_size=min(512, max(64, 2048 // grid_size)),
        n_steps=max(512, 4096 // grid_size),
        ent_coef=ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu",
    )

    print(f"Curriculum settings:")
    print(f"  batch_size: {model.batch_size}")
    print(f"  n_steps: {model.n_steps}")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  ent_coef: {model.ent_coef}")
    print(f"  net_arch: {net_arch}")
    print(f"  n_envs: {n_envs}")

    # Quick training with same timesteps as train.py style
    model.learn(total_timesteps=50000)

    # Test
    test_env = make_curriculum_env()
    success_count = 0

    for episode in range(10):
        obs, _ = test_env.reset()
        for step in range(test_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            if terminated:
                success_count += 1
                break

    success_rate = success_count / 10
    print(f"Curriculum style success rate: {success_rate:.1%}")

    env.close()
    return success_rate


def test_environment_consistency():
    """Test if both approaches create the same environment"""
    print("\n🔍 Testing environment consistency...")

    # Train.py style env
    env1 = make_env()

    # Curriculum style env
    env2 = make_env(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS)
    env2.max_steps = GRID_SIZE * GRID_SIZE * 4 + NUM_OBJECTS * 50

    print(f"Train.py env max_steps: {env1.max_steps}")
    print(f"Curriculum env max_steps: {env2.max_steps}")
    print(f"Grid sizes match: {env1.grid_size == env2.grid_size}")
    print(f"Num objects match: {env1.num_objects == env2.num_objects}")
    print(f"Use shaping match: {env1.use_shaping == env2.use_shaping}")


if __name__ == "__main__":
    print("🐛 Debugging curriculum vs train.py performance gap")
    print("=" * 50)

    # Test environment consistency first
    test_environment_consistency()

    # Test both approaches
    train_success = test_train_py_style()
    curriculum_success = test_curriculum_style()

    print(f"\n📊 RESULTS COMPARISON:")
    print(f"Train.py style:    {train_success:.1%}")
    print(f"Curriculum style:  {curriculum_success:.1%}")
    print(f"Performance gap:   {abs(train_success - curriculum_success):.1%}")

    if curriculum_success < train_success * 0.5:
        print("\n🚨 SIGNIFICANT PERFORMANCE GAP DETECTED!")
        print("Likely causes:")
        print("1. Hyperparameter mismatch (batch_size, n_steps)")
        print("2. SubprocVecEnv vs DummyVecEnv difference")
        print("3. Environment creation differences")
        print("4. Network architecture differences")
