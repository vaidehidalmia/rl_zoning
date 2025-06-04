# Deep debugging to find the real issue

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from config import *
from utils import make_env


def analyze_environment_behavior():
    """Deep dive into environment behavior"""
    print("🔬 DEEP ENVIRONMENT ANALYSIS")
    print("=" * 50)

    env = make_env()
    print(f"Environment created with:")
    print(f"  Grid size: {env.grid_size}")
    print(f"  Num objects: {env.num_objects}")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Use shaping: {env.use_shaping}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space: {env.observation_space}")

    # Test multiple episodes to understand the task
    print(f"\n📊 Analyzing {env.num_objects} object(s) placement...")

    for episode in range(5):
        obs, _ = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Agent start: {env.agent_pos}")
        print(f"  Objects: {env.objects}")

        # Check if objects are actually in wrong zones
        for pos, obj_type in env.objects.items():
            r, c = pos
            is_correct = env.is_correct_zone(pos, obj_type)
            zone = "red" if c < env.grid_size // 2 else "blue"
            obj_name = "red" if obj_type == 1 else "blue"
            print(
                f"    {obj_name} object at {pos} in {zone} zone - {'✅ CORRECT' if is_correct else '❌ WRONG'}"
            )

        # Check completion condition
        complete = env.is_episode_complete()
        print(f"  Task complete: {complete}")

        if complete:
            print("  🚨 TASK STARTS COMPLETED! This is the bug!")


def test_manual_solution():
    """Test if we can manually solve the task"""
    print(f"\n🎯 MANUAL SOLUTION TEST")
    print("=" * 50)

    env = make_env()
    obs, _ = env.reset()

    print(f"Initial state:")
    print(f"  Agent: {env.agent_pos}")
    print(f"  Objects: {env.objects}")
    print(f"  Carried: {env.carried_object}")

    # Manual solution strategy
    steps = 0
    max_manual_steps = 100

    while steps < max_manual_steps:
        # Check if task is complete
        if env.is_episode_complete():
            print(f"✅ MANUALLY SOLVED in {steps} steps!")
            return True

        # Simple strategy: pick up first object, move to correct zone, drop
        if env.carried_object == -1:
            # Need to pick up an object
            if env.agent_pos in env.objects:
                action = 4  # pickup
                print(f"Step {steps}: Pickup at {env.agent_pos}")
            else:
                # Move to nearest object
                if env.objects:
                    target_pos = next(iter(env.objects.keys()))
                    action = get_move_action(env.agent_pos, target_pos)
                    print(f"Step {steps}: Moving toward object at {target_pos}")
                else:
                    print("No objects to pick up!")
                    break
        else:
            # Carrying object, need to find correct zone
            obj_type = env.carried_object
            target_zone = (
                0 if obj_type == 1 else env.grid_size - 1
            )  # red=left, blue=right
            target_pos = (env.agent_pos[0], target_zone)

            if env.agent_pos[1] == target_zone:
                # In correct zone, drop it
                action = 5  # drop
                print(f"Step {steps}: Dropping {obj_type} object in correct zone")
            else:
                # Move to correct zone
                action = get_move_action(env.agent_pos, target_pos)
                print(f"Step {steps}: Moving to correct zone for object {obj_type}")

        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        print(
            f"  Agent: {env.agent_pos}, Carried: {env.carried_object}, Reward: {reward:.2f}"
        )

        if terminated:
            print(f"✅ TASK COMPLETED in {steps} steps!")
            return True
        elif truncated:
            print(f"❌ Episode truncated after {steps} steps")
            return False

    print(f"❌ Manual solution failed after {max_manual_steps} steps")
    return False


def get_move_action(current_pos, target_pos):
    """Get action to move toward target"""
    r, c = current_pos
    tr, tc = target_pos

    if r > tr:
        return 0  # up
    elif r < tr:
        return 1  # down
    elif c > tc:
        return 2  # left
    elif c < tc:
        return 3  # right
    else:
        return 4  # at target, try pickup


def test_reward_structure():
    """Analyze the reward structure"""
    print(f"\n💰 REWARD STRUCTURE ANALYSIS")
    print("=" * 50)

    env = make_env()
    obs, _ = env.reset()

    print(f"Reward values:")
    print(f"  Step cost: {env.rewards.STEP_COST}")
    print(f"  Invalid action: {env.rewards.INVALID_ACTION}")
    print(f"  Task complete: {env.rewards.TASK_COMPLETE}")

    # Test different actions and their rewards
    print(f"\nTesting action rewards:")

    # Test movement
    initial_pos = env.agent_pos
    for action in range(4):
        env_copy = make_env()
        env_copy.reset()
        env_copy.agent_pos = initial_pos
        env_copy.objects = env.objects.copy()

        obs, reward, _, _, _ = env_copy.step(action)
        print(f"  Action {action} (move): reward = {reward:.2f}")

    # Test pickup when no object
    env_copy = make_env()
    env_copy.reset()
    env_copy.agent_pos = (0, 0)  # Position with no object
    env_copy.objects = {(1, 1): 1}  # Object elsewhere

    obs, reward, _, _, _ = env_copy.step(4)  # pickup
    print(f"  Action 4 (pickup, no object): reward = {reward:.2f}")

    # Test pickup with object
    env_copy = make_env()
    env_copy.reset()
    env_copy.agent_pos = (1, 1)
    env_copy.objects = {(1, 1): 1}  # Object at agent position

    obs, reward, _, _, _ = env_copy.step(4)  # pickup
    print(f"  Action 4 (pickup, with object): reward = {reward:.2f}")


def test_minimal_training():
    """Test with minimal, well-tuned hyperparameters"""
    print(f"\n🎯 MINIMAL TRAINING TEST")
    print("=" * 50)

    def make_simple_env():
        return make_env()

    # Single environment, simple hyperparameters
    env = make_vec_env(make_simple_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Larger rollout for better learning
        batch_size=64,  # Smaller batch size
        n_epochs=10,  # More gradient steps
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Very low entropy for exploitation
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[64, 64]  # Smaller network for simple task
        ),
        device="cpu",
    )

    print("Training with minimal hyperparameters...")
    model.learn(total_timesteps=100000)

    # Test the trained model
    test_env = make_env()
    success_count = 0

    for episode in range(20):
        obs, _ = test_env.reset()
        for step in range(test_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            if terminated:
                success_count += 1
                break

    success_rate = success_count / 20
    print(f"Minimal training success rate: {success_rate:.1%}")

    env.close()
    return success_rate


def analyze_observation_space():
    """Analyze what the agent actually sees"""
    print(f"\n👁️ OBSERVATION SPACE ANALYSIS")
    print("=" * 50)

    env = make_env()
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation channels:")
    print(f"  Channel 0: Object positions and types")
    print(f"  Channel 1: Zone definitions")
    print(f"  Channel 2: Agent position and carried object")

    print(f"\nCurrent observation:")
    for i in range(3):
        print(f"Channel {i}:")
        print(obs[:, :, i])
        print()


if __name__ == "__main__":
    print("🐛 DEEP DEBUGGING ANALYSIS")
    print("=" * 60)

    # Step 1: Analyze environment
    analyze_environment_behavior()

    # Step 2: Test manual solution
    manual_success = test_manual_solution()

    # Step 3: Analyze rewards
    test_reward_structure()

    # Step 4: Analyze observations
    analyze_observation_space()

    # Step 5: Test minimal training
    if manual_success:
        print("\n" + "=" * 60)
        minimal_success = test_minimal_training()

        if minimal_success > 0.5:
            print("✅ Minimal training works! The issue is hyperparameters.")
        else:
            print("❌ Even minimal training fails. Deeper environment issue.")
    else:
        print("❌ Manual solution failed. Environment has fundamental issues.")
        print("Check the task setup - objects might start in correct positions!")
