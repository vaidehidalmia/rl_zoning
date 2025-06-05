# Fixed Video Recorder - Compatible with modern matplotlib

import numpy as np
import os
import json
import imageio
from stable_baselines3 import PPO
import glob

from config import *
from utils import make_env, load_model_safe


def find_trained_models():
    """Find all trained models from the curriculum"""
    print("🔍 FINDING TRAINED MODELS")
    print("=" * 60)

    trained_models = []

    # Look for stage models
    stage_pattern = f"{MODELS_DIR}/stage_*"
    stage_dirs = glob.glob(stage_pattern)

    for stage_dir in sorted(stage_dirs):
        # Extract stage info from directory name
        dir_name = os.path.basename(stage_dir)
        # Example: stage_1_grid_4x4_obj_1

        try:
            parts = dir_name.split("_")
            stage_num = int(parts[1])

            # Parse grid size correctly - handle "4x4" format
            grid_part = parts[3]  # Should be "4x4"
            if "x" in grid_part:
                grid_size = int(grid_part.split("x")[0])
            else:
                continue

            num_objects = int(parts[5])

            # Look for best model first, then final model
            best_model_path = f"{stage_dir}/best/best_model"
            final_model_path = f"{stage_dir}/final"

            if os.path.exists(best_model_path + ".zip"):
                model_path = best_model_path
                model_type = "best"
            elif os.path.exists(final_model_path + ".zip"):
                model_path = final_model_path
                model_type = "final"
            else:
                print(f"   ⚠️  No model found in {stage_dir}")
                continue

            trained_models.append(
                {
                    "stage": stage_num,
                    "grid_size": grid_size,
                    "num_objects": num_objects,
                    "model_path": model_path,
                    "model_type": model_type,
                    "stage_key": f"{grid_size}x{grid_size}_{num_objects}obj",
                }
            )

            print(
                f"   ✅ Stage {stage_num}: {grid_size}x{grid_size}, {num_objects} obj ({model_type} model)"
            )

        except (ValueError, IndexError) as e:
            print(f"   ❌ Could not parse {dir_name}: {e}")

    print(f"\n📊 Found {len(trained_models)} trained models")
    return sorted(trained_models, key=lambda x: (x["grid_size"], x["num_objects"]))


def test_rendering_compatibility():
    """Test if rendering works before recording videos"""
    print("🧪 Testing rendering compatibility...")

    try:
        # Test with smallest grid first
        env = make_env(grid_size=4, num_objects=1, render_mode="rgb_array")
        obs, _ = env.reset()

        # Try to render
        frame = env.render()

        if frame is not None and isinstance(frame, np.ndarray):
            print("✅ Rendering works!")
            print(f"   Frame shape: {frame.shape}")
            env.close()
            return True
        else:
            print("❌ Rendering returned None or invalid data")
            env.close()
            return False

    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        print("\n💡 Possible fixes:")
        print("1. Update matplotlib: pip install matplotlib --upgrade")
        print("2. Check visualize/renderer.py compatibility")
        print("3. Use headless mode: export MPLBACKEND=Agg")
        return False


def create_simple_text_frame(text, width=400, height=400):
    """Create a simple colored frame with text info (fallback)"""
    # Create a colored frame based on the text
    if "success" in text.lower():
        color = [0, 255, 0]  # Green
    elif "timeout" in text.lower():
        color = [255, 165, 0]  # Orange
    else:
        color = [100, 100, 100]  # Gray

    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def record_model_videos_safe(model_info, episodes=VIDEO_EPISODES):
    """Record videos with error handling and fallbacks"""
    stage = model_info["stage"]
    grid_size = model_info["grid_size"]
    num_objects = model_info["num_objects"]
    model_path = model_info["model_path"]
    model_type = model_info["model_type"]

    print(
        f"\n🎬 Recording videos for {model_info['stage_key']} ({model_type} model)..."
    )

    # Create recordings directory
    if isinstance(stage, int):
        video_dir = (
            f"{RECORDINGS_DIR}/stage_{stage}_{grid_size}x{grid_size}_{num_objects}obj"
        )
    else:
        video_dir = f"{RECORDINGS_DIR}/{stage}"

    os.makedirs(video_dir, exist_ok=True)

    # Load model
    model = load_model_safe(model_path)
    if model is None:
        print(f"❌ Could not load model: {model_path}")
        return False

    print(f"📁 Saving videos to: {video_dir}")

    success_count = 0
    episode_data = []
    rendering_failed_episodes = 0

    for episode in range(episodes):
        try:
            # Create environment with rendering
            env = make_env(
                grid_size=grid_size, num_objects=num_objects, render_mode="rgb_array"
            )

            frames = []
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_rendering_failed = False

            # Render initial state
            try:
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    frames.append(frame.copy())  # Make a copy to avoid reference issues
                else:
                    episode_rendering_failed = True
            except Exception as e:
                if episode == 0:  # Only warn on first episode
                    print(f"   ⚠️  Initial rendering failed: {e}")
                episode_rendering_failed = True

            # Run episode and collect frames for EVERY step
            for step in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Try to record frame for this step (don't give up on rendering for the whole episode)
                if not episode_rendering_failed:
                    try:
                        frame = env.render()
                        if frame is not None and isinstance(frame, np.ndarray):
                            frames.append(frame.copy())
                        else:
                            # Only mark as failed if we get None consistently
                            episode_rendering_failed = True
                    except Exception as e:
                        if step == 0:  # Only warn on first step failure
                            print(f"   ⚠️  Step rendering failed: {e}")
                        episode_rendering_failed = True

                if terminated or truncated:
                    break

            # Render final state if episode ended early
            if (terminated or truncated) and not episode_rendering_failed:
                try:
                    frame = env.render()
                    if frame is not None and isinstance(frame, np.ndarray):
                        frames.append(frame.copy())
                except:
                    pass  # Don't fail the whole episode for final frame

            # Determine episode outcome
            outcome = "success" if terminated else "timeout"
            if terminated:
                success_count += 1

            episode_info = {
                "episode": episode + 1,
                "outcome": outcome,
                "reward": round(episode_reward, 1),
                "steps": episode_steps,
                "success": terminated,
                "frames_collected": len(frames),
            }
            episode_data.append(episode_info)

            # Save video or create fallback
            filename = f"ep{episode + 1:02d}_{outcome}_r{episode_reward:.1f}_s{episode_steps:03d}"

            if (
                frames and len(frames) > 1
            ):  # Need at least 2 frames for a meaningful video
                # Save actual video
                if VIDEO_FORMAT == "gif":
                    video_path = f"{video_dir}/{filename}.gif"
                    try:
                        # Ensure all frames have the same shape
                        if len(set(frame.shape for frame in frames)) == 1:
                            imageio.mimsave(
                                video_path, frames, duration=VIDEO_DURATION, loop=0
                            )
                            print(f"   ✅ {filename}.gif ({len(frames)} frames)")
                        else:
                            print(f"   ⚠️  Frame shape mismatch, skipping {filename}")
                            episode_rendering_failed = True
                    except Exception as e:
                        print(f"   ❌ Failed to save GIF: {e}")
                        episode_rendering_failed = True

                elif VIDEO_FORMAT == "mp4":
                    video_path = f"{video_dir}/{filename}.mp4"
                    try:
                        # Ensure all frames have the same shape
                        if len(set(frame.shape for frame in frames)) == 1:
                            imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
                            print(f"   ✅ {filename}.mp4 ({len(frames)} frames)")
                        else:
                            print(f"   ⚠️  Frame shape mismatch, skipping {filename}")
                            episode_rendering_failed = True
                    except Exception as e:
                        print(f"   ❌ Failed to save MP4: {e}")
                        episode_rendering_failed = True
            else:
                episode_rendering_failed = True

            # Create fallback summary if rendering failed
            if episode_rendering_failed:
                rendering_failed_episodes += 1
                if episode == 0:
                    print(f"   📊 Creating episode summaries (rendering issues)")

                # Create a simple info frame
                fallback_frame = create_simple_text_frame(
                    f"{outcome}: r={episode_reward:.1f}, steps={episode_steps}"
                )

                if VIDEO_FORMAT == "gif":
                    video_path = f"{video_dir}/{filename}_summary.gif"
                    try:
                        # Create a simple 1-frame GIF
                        imageio.mimsave(video_path, [fallback_frame], duration=2.0)
                    except:
                        pass

            env.close()

        except Exception as e:
            print(f"   ❌ Episode {episode + 1} failed: {e}")
            continue

    # Save episode summary
    success_rate = success_count / episodes
    rendering_success_rate = (episodes - rendering_failed_episodes) / episodes

    summary = {
        "model_info": model_info,
        "success_rate": success_rate,
        "rendering_success_rate": rendering_success_rate,
        "episodes": episode_data,
        "video_settings": {
            "format": VIDEO_FORMAT,
            "duration": VIDEO_DURATION if VIDEO_FORMAT == "gif" else None,
            "fps": VIDEO_FPS if VIDEO_FORMAT == "mp4" else None,
            "total_episodes": episodes,
            "rendering_failed_episodes": rendering_failed_episodes,
        },
    }

    summary_path = f"{video_dir}/episode_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   📊 Success rate: {success_rate:.1%} ({success_count}/{episodes})")
    print(
        f"   🎬 Rendering success: {rendering_success_rate:.1%} ({episodes - rendering_failed_episodes}/{episodes})"
    )
    print(f"   📄 Summary saved: episode_summary.json")

    return True


def create_text_summary_report():
    """Create a text summary of all model performances"""
    print(f"\n📄 CREATING TEXT SUMMARY REPORT")
    print("=" * 60)

    trained_models = find_trained_models()
    if not trained_models:
        print("❌ No trained models found")
        return

    summary_dir = f"{RECORDINGS_DIR}/summary"
    os.makedirs(summary_dir, exist_ok=True)

    report_lines = []
    report_lines.append("🎓 CURRICULUM PERFORMANCE REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")

    total_tests = 0
    overall_performance = []

    for model_info in trained_models:
        stage = model_info["stage"]
        grid_size = model_info["grid_size"]
        num_objects = model_info["num_objects"]
        model_path = model_info["model_path"]
        stage_key = model_info["stage_key"]

        print(f"   📊 Testing {stage_key}...")

        # Load model and test performance
        model = load_model_safe(model_path)
        if model is None:
            continue

        success_count = 0
        total_reward = 0
        total_steps = 0
        episodes = 20  # Test episodes

        for episode in range(episodes):
            try:
                env = make_env(grid_size=grid_size, num_objects=num_objects)
                obs, _ = env.reset()
                episode_reward = 0

                for step in range(env.max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        if terminated:
                            success_count += 1
                        total_steps += step + 1
                        break
                else:
                    total_steps += env.max_steps

                total_reward += episode_reward
                env.close()

            except Exception as e:
                print(f"     ❌ Episode {episode + 1} failed: {e}")
                continue

        success_rate = success_count / episodes
        avg_reward = total_reward / episodes
        avg_steps = total_steps / episodes

        overall_performance.append(success_rate)
        total_tests += 1

        # Add to report
        status = (
            "🏆"
            if success_rate >= 0.8
            else "✅"
            if success_rate >= 0.6
            else "⚠️"
            if success_rate >= 0.4
            else "❌"
        )

        report_lines.append(f"{status} Stage {stage}: {stage_key}")
        report_lines.append(f"   Success Rate: {success_rate:.1%}")
        report_lines.append(f"   Avg Reward: {avg_reward:.1f}")
        report_lines.append(f"   Avg Steps: {avg_steps:.1f}")
        report_lines.append(f"   Model: {model_info['model_type']}")
        report_lines.append("")

    # Overall summary
    if overall_performance:
        overall_success = sum(overall_performance) / len(overall_performance)
        excellent_stages = sum(1 for p in overall_performance if p >= 0.8)
        good_stages = sum(1 for p in overall_performance if 0.6 <= p < 0.8)

        report_lines.append("📊 OVERALL CURRICULUM PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Success Rate: {overall_success:.1%}")
        report_lines.append(
            f"Excellent Stages (≥80%): {excellent_stages}/{total_tests}"
        )
        report_lines.append(f"Good Stages (≥60%): {good_stages}/{total_tests}")
        report_lines.append("")
        report_lines.append("🎯 CURRICULUM INSIGHTS:")

        if excellent_stages >= total_tests // 2:
            report_lines.append("✅ Curriculum is working well!")
        elif good_stages >= total_tests // 2:
            report_lines.append("⚠️  Curriculum shows promise but needs optimization")
        else:
            report_lines.append("❌ Curriculum needs significant improvement")

    # Save report
    report_path = f"{summary_dir}/curriculum_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Report saved: {report_path}")

    # Print summary to console
    print("\n📊 QUICK SUMMARY:")
    for line in report_lines[-10:]:  # Last 10 lines
        print(line)


def record_all_trained_models_safe():
    """Record videos for all trained models with error handling"""
    print("🎬 VIDEO RECORDER FOR TRAINED MODELS (SAFE MODE)")
    print("=" * 60)

    # Test rendering first
    if not test_rendering_compatibility():
        print("\n⚠️  Rendering issues detected. Creating text summaries instead.")
        create_text_summary_report()
        return

    # Ensure recordings directory exists
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # Find all trained models
    trained_models = find_trained_models()

    if not trained_models:
        print("❌ No trained models found!")
        print("   Make sure you have run the curriculum training first.")
        return

    print(f"\n🎬 Recording videos for {len(trained_models)} models...")
    print(f"📹 Video format: {VIDEO_FORMAT}")
    print(f"📁 Output directory: {RECORDINGS_DIR}")

    successful_recordings = 0

    for i, model_info in enumerate(trained_models, 1):
        print(f"\n[{i}/{len(trained_models)}] {model_info['stage_key']}")

        if record_model_videos_safe(model_info):
            successful_recordings += 1

    print(f"\n📊 RECORDING SUMMARY")
    print("=" * 40)
    print(
        f"Successfully recorded: {successful_recordings}/{len(trained_models)} models"
    )
    print(f"Videos saved to: {RECORDINGS_DIR}/")

    # Also create text summary
    create_text_summary_report()

    print(f"\n🎯 CHECK RESULTS:")
    print(f"1. Videos: {RECORDINGS_DIR}/stage_*/")
    print(f"2. Summaries: {RECORDINGS_DIR}/stage_*/episode_summary.json")
    print(f"3. Report: {RECORDINGS_DIR}/summary/curriculum_report.txt")

    return successful_recordings


if __name__ == "__main__":
    print("🎬 FIXED VIDEO RECORDER FOR TRAINED MODELS")
    print("=" * 60)

    record_all_trained_models_safe()
