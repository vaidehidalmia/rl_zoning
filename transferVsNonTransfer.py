# Fixed Transfer Learning - Consistent Network Architectures

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os


def calculate_training_params_consistent(grid_size, num_objects, has_previous=False):
    """FIXED: Consistent network architectures for transfer learning"""
    complexity = grid_size * grid_size * num_objects

    # KEY FIX: Use consistent network architecture for transfer compatibility
    if complexity <= 50:  # Both 4x4,1obj and 4x4,2obj use same network!
        base_timesteps = (
            200_000 if num_objects == 1 else (600_000 if has_previous else 800_000)
        )
        learning_rate = 3e-4 if num_objects == 1 else 2e-4
        net_arch = [128, 128]  # CONSISTENT! Same for both single and multi-object
        ent_coef = 0.05 if num_objects == 1 else 0.08
    elif complexity <= 100:  # 5x5 problems
        base_timesteps = (
            300_000 if num_objects == 1 else (800_000 if has_previous else 1_200_000)
        )
        learning_rate = 3e-4 if num_objects == 1 else 1e-4
        net_arch = [256, 256]  # CONSISTENT for 5x5 problems
        ent_coef = 0.05 if num_objects == 1 else 0.1
    else:  # Very complex
        base_timesteps = 2_000_000
        learning_rate = 1e-4
        net_arch = [512, 256]
        ent_coef = 0.1

    timesteps = min(base_timesteps, 2_000_000)
    max_steps = grid_size * grid_size * 4 + num_objects * 50

    return timesteps, max_steps, learning_rate, net_arch, ent_coef


def smart_transfer_weights(source_model, target_model):
    """Smart weight transfer that handles different architectures"""
    try:
        # Try direct transfer first
        target_model.policy.load_state_dict(source_model.policy.state_dict())
        print("✅ Direct weight transfer successful!")
        return True
    except Exception as e:
        print(f"⚠️  Direct transfer failed: {e}")

        # Try partial transfer (transfer what we can)
        try:
            source_dict = source_model.policy.state_dict()
            target_dict = target_model.policy.state_dict()

            transferred_layers = 0
            for key in target_dict.keys():
                if (
                    key in source_dict
                    and source_dict[key].shape == target_dict[key].shape
                ):
                    target_dict[key] = source_dict[key]
                    transferred_layers += 1

            target_model.policy.load_state_dict(target_dict)
            print(
                f"✅ Partial transfer successful! Transferred {transferred_layers} layers"
            )
            return True

        except Exception as e2:
            print(f"❌ Partial transfer also failed: {e2}")
            return False


def train_stage_fixed_transfer(stage, grid_size, num_objects, previous_model_path=None):
    """Fixed transfer learning with consistent architectures"""
    print(
        f"\n🎓 CURRICULUM STAGE {stage}: {grid_size}x{grid_size} Grid, {num_objects} Objects"
    )
    print("=" * 60)

    # Get CONSISTENT training parameters
    timesteps, max_steps, learning_rate, net_arch, ent_coef = (
        calculate_training_params_consistent(
            grid_size, num_objects, has_previous=previous_model_path is not None
        )
    )

    print(f"📊 Training metrics:")
    print(f"   Complexity: {grid_size * grid_size * num_objects}")
    print(f"   Training steps: {timesteps:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Network: {net_arch}")
    print(f"   Exploration: {ent_coef}")

    # Environment setup
    def make_curriculum_env():
        from utils import make_env

        return make_env(grid_size=grid_size, num_objects=num_objects)

    complexity = grid_size * grid_size * num_objects
    n_envs = 1 if complexity <= 25 else 2
    env = make_vec_env(make_curriculum_env, n_envs=n_envs)

    # PPO parameters
    ppo_params = {
        "n_steps": 2048 if num_objects == 1 else 4096,
        "batch_size": 64 if num_objects == 1 else 128,
        "n_epochs": 10 if num_objects == 1 else 15,
        "gamma": 0.99 if num_objects == 1 else 0.995,
        "gae_lambda": 0.95 if num_objects == 1 else 0.98,
        "clip_range": 0.2 if num_objects == 1 else 0.1,
        "vf_coef": 0.5 if num_objects == 1 else 1.0,
        "max_grad_norm": 0.5,
    }

    # Create target model
    target_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
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
            transfer_successful = smart_transfer_weights(source_model, target_model)

            if transfer_successful:
                # Fine-tune with lower learning rate
                target_model.learning_rate = learning_rate * 0.5
                print(
                    f"🔧 Reduced learning rate to {target_model.learning_rate} for fine-tuning"
                )

        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")

    if not transfer_successful:
        print("🆕 Training from scratch")

    # Training
    transfer_note = "with transfer learning" if transfer_successful else "from scratch"
    print(f"🚀 Training {transfer_note} for {timesteps:,} timesteps...")

    try:
        target_model.learn(total_timesteps=timesteps)

        # Save model
        from utils import get_model_path

        stage_dir = get_model_path(grid_size, num_objects, stage)
        os.makedirs(stage_dir, exist_ok=True)
        stage_path = f"{stage_dir}/final"
        target_model.save(stage_path)
        print(f"✅ Saved model: {stage_path}")

        env.close()
        return stage_path

    except Exception as e:
        print(f"❌ Training failed: {e}")
        env.close()
        return None


def evaluate_model_quick(model_path, grid_size, num_objects, episodes=15):
    """Quick evaluation"""
    from utils import load_model_safe, make_env

    model = load_model_safe(model_path)
    if model is None:
        return 0.0, 0.0, 0.0

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
    avg_reward = sum(total_rewards) / episodes
    avg_length = sum(episode_lengths) / episodes

    return success_rate, avg_reward, avg_length


def test_fixed_transfer():
    """Test the fixed transfer learning approach"""
    print("🧪 TESTING FIXED TRANSFER LEARNING")
    print("=" * 60)

    from utils import create_directories

    create_directories()

    # Test case: 4x4 with 1 → 2 objects
    print("Testing 4x4: 1 object → 2 objects with CONSISTENT architectures")

    # Step 1: Train base model (4x4, 1 object)
    print("\n1️⃣ Training base model (4x4, 1 object)...")
    base_model_path = train_stage_fixed_transfer(1, 4, 1, None)

    if not base_model_path:
        print("❌ Base model training failed")
        return

    # Evaluate base model
    base_success, _, _ = evaluate_model_quick(base_model_path, 4, 1, 10)
    print(f"   Base model success: {base_success:.1%}")

    if base_success < 0.8:
        print("⚠️  Base model not good enough for transfer")
        return

    # Step 2: Train target model WITH transfer
    print("\n2️⃣ Training target model WITH transfer (4x4, 2 objects)...")
    transfer_model_path = train_stage_fixed_transfer(2, 4, 2, base_model_path)

    if not transfer_model_path:
        print("❌ Transfer model training failed")
        return

    # Evaluate transfer model
    transfer_success, transfer_reward, transfer_steps = evaluate_model_quick(
        transfer_model_path, 4, 2, 20
    )

    # Step 3: Train target model WITHOUT transfer (for comparison)
    print("\n3️⃣ Training target model WITHOUT transfer (4x4, 2 objects)...")
    fresh_model_path = train_stage_fixed_transfer(3, 4, 2, None)

    if fresh_model_path:
        fresh_success, fresh_reward, fresh_steps = evaluate_model_quick(
            fresh_model_path, 4, 2, 20
        )
    else:
        fresh_success, fresh_reward, fresh_steps = 0.0, 0.0, 0.0

    # Results
    print(f"\n🏆 FIXED TRANSFER LEARNING RESULTS:")
    print(f"=" * 50)
    print(
        f"WITH transfer:    {transfer_success:5.1%} success | {transfer_reward:6.1f} reward | {transfer_steps:5.1f} steps"
    )
    print(
        f"WITHOUT transfer: {fresh_success:5.1%} success | {fresh_reward:6.1f} reward | {fresh_steps:5.1f} steps"
    )
    print(f"Improvement:      {transfer_success - fresh_success:5.1%}")

    if transfer_success > fresh_success + 0.15:
        print("🎉 EXCELLENT! Transfer learning shows major improvement!")
    elif transfer_success > fresh_success + 0.05:
        print("✅ GOOD! Transfer learning shows improvement!")
    elif transfer_success > fresh_success:
        print("✅ Transfer learning shows modest improvement!")
    else:
        print("⚠️  Transfer learning needs more tuning")

    return transfer_success, fresh_success


def create_consistent_curriculum():
    """Create a curriculum with consistent architectures for transfer"""
    print("📚 CONSISTENT ARCHITECTURE CURRICULUM")
    print("=" * 60)

    # Curriculum with consistent architectures within complexity tiers
    curriculum = [
        # Tier 1: Use [128, 128] for both
        (4, 1),  # Complexity 16 - but use [128, 128] for transfer compatibility
        (4, 2),  # Complexity 32 - use [128, 128]
        # Tier 2: Use [256, 256] for both
        (5, 1),  # Complexity 25 - use [256, 256] for transfer compatibility
        (5, 2),  # Complexity 50 - use [256, 256]
    ]

    print("Architecture plan:")
    for stage, (grid_size, num_objects) in enumerate(curriculum, 1):
        complexity = grid_size * grid_size * num_objects
        arch = "[128, 128]" if complexity <= 50 else "[256, 256]"
        print(f"   Stage {stage}: {grid_size}x{grid_size}, {num_objects} obj → {arch}")

    return curriculum


if __name__ == "__main__":
    print("🔧 FIXED TRANSFER LEARNING")
    print("=" * 60)

    print("Key fixes:")
    print("✅ Consistent network architectures within complexity tiers")
    print("✅ Smart weight transfer with fallback to partial transfer")
    print("✅ Proper fine-tuning with reduced learning rate")
    print()

    choice = input(
        "Choose: [1] Test fixed transfer, [2] Show curriculum plan: "
    ).strip()

    if choice == "2":
        create_consistent_curriculum()
    else:
        test_fixed_transfer()
