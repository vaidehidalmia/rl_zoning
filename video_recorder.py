import os
import numpy as np
import imageio
from config import *
from utils import make_env, load_model_safe, create_directories

# Simple matplotlib-based renderer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SimpleRenderer:
    """Simple renderer for video recording"""

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def render(self, obs):
        """Create visualization from observation"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Agent Performance ({self.grid_size}x{self.grid_size})", fontweight="bold"
        )

        # Draw zones
        # Red zone (left)
        red_zone = patches.Rectangle(
            (-0.5, -0.5),
            self.grid_size // 2,
            self.grid_size,
            facecolor="red",
            alpha=0.2,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(red_zone)

        # Blue zone (right)
        blue_zone = patches.Rectangle(
            (self.grid_size // 2 - 0.5, -0.5),
            self.grid_size // 2,
            self.grid_size,
            facecolor="blue",
            alpha=0.2,
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(blue_zone)

        # Find agent and objects from observation
        agent_pos = None
        carried_obj = -1

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Objects (channel 0)
                obj_type = obs[r, c, 0]
                if obj_type > 0:
                    color = "red" if obj_type == 1 else "blue"
                    obj_rect = patches.Rectangle(
                        (c - 0.3, r - 0.3),
                        0.6,
                        0.6,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=2,
                        alpha=0.8,
                    )
                    ax.add_patch(obj_rect)
                    ax.text(
                        c,
                        r,
                        str(obj_type),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white",
                    )

                # Agent (channel 2)
                agent_indicator = obs[r, c, 2]
                if agent_indicator > 0:
                    agent_pos = (r, c)
                    if agent_indicator > 10:
                        carried_obj = agent_indicator - 10

        # Draw agent
        if agent_pos:
            r, c = agent_pos
            agent_circle = patches.Circle(
                (c, r), 0.25, facecolor="yellow", edgecolor="black", linewidth=3
            )
            ax.add_patch(agent_circle)
            ax.text(c, r, "A", ha="center", va="center", fontsize=12, fontweight="bold")

            if carried_obj > 0:
                ax.text(
                    c,
                    r + 0.4,
                    f"Carrying: {carried_obj}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        # Convert to array - use modern matplotlib API
        fig.canvas.draw()

        # Try multiple methods for cross-platform compatibility
        try:
            # Method 1: Modern buffer_rgba approach
            buf = np.asarray(fig.canvas.buffer_rgba())
            height, width, _ = buf.shape
            buf = buf[:, :, :3]  # Convert RGBA to RGB
        except:
            try:
                # Method 2: Alternative buffer method
                width, height = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(height, width, 4)[:, :, :3]  # RGBA to RGB
            except:
                # Method 3: Fallback using PIL
                import io
                from PIL import Image

                buf_io = io.BytesIO()
                fig.savefig(
                    buf_io,
                    format="png",
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                    dpi=100,
                )
                buf_io.seek(0)
                img = Image.open(buf_io)
                buf = np.array(img)
                if buf.shape[2] == 4:  # RGBA to RGB
                    buf = buf[:, :, :3]

        plt.close(fig)
        return buf


def create_agent_video():
    """Create video of agent performance using config settings"""

    # Load model using util function
    model = load_model_safe()
    if model is None:
        return

    print(f"✅ Loaded model: {MODEL_PATH}")

    # Create output directory using util function
    create_directories()

    # Create renderer
    renderer = SimpleRenderer(GRID_SIZE)

    print(f"🎬 Recording {VIDEO_EPISODES} episodes...")

    for episode in range(VIDEO_EPISODES):
        print(f"Recording episode {episode + 1}...")

        # Create environment using util function
        env = make_env()
        obs, _ = env.reset()

        frames = []

        # Add initial frame
        frame = renderer.render(env.get_obs())
        frames.append(frame)

        episode_reward = 0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Capture frame
            frame = renderer.render(env.get_obs())
            frames.append(frame)

            if terminated or truncated:
                status = "COMPLETED" if terminated else "TIMEOUT"
                print(
                    f"  Episode {episode + 1}: {status} in {step + 1} steps (reward: {episode_reward:.1f})"
                )
                break

        # Save video
        filename = f"{RECORDINGS_DIR}/episode_{episode + 1}.{VIDEO_FORMAT}"

        if VIDEO_FORMAT.lower() == "gif":
            imageio.mimsave(filename, frames, duration=VIDEO_DURATION, loop=0)
        elif VIDEO_FORMAT.lower() == "mp4":
            try:
                imageio.mimsave(filename, frames, fps=VIDEO_FPS)
            except Exception as e:
                print(f"MP4 save failed: {e}, saving as GIF instead")
                gif_filename = f"{RECORDINGS_DIR}/episode_{episode + 1}.gif"
                imageio.mimsave(gif_filename, frames, duration=VIDEO_DURATION, loop=0)
                filename = gif_filename

        print(f"  💾 Saved: {filename}")

    print(f"\n🎯 All videos saved to {RECORDINGS_DIR}/")


def create_comparison_video():
    """Create side-by-side comparison of two episodes"""

    model = load_model_safe()
    if model is None:
        print("❌ No trained model found!")
        return

    create_directories()
    renderer = SimpleRenderer(GRID_SIZE)

    print("🎬 Creating comparison video...")

    all_frames = []

    # Record 2 episodes
    for episode in range(2):
        env = make_env()
        obs, _ = env.reset()

        frames = []
        frame = renderer.render(env.get_obs())
        frames.append(frame)

        for step in range(50):  # Shorter for comparison
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            frame = renderer.render(env.get_obs())
            frames.append(frame)

            if terminated or truncated:
                break

        all_frames.append(frames)

    # Create side-by-side frames
    if len(all_frames) == 2:
        from PIL import Image

        max_frames = max(len(all_frames[0]), len(all_frames[1]))
        combined_frames = []

        for i in range(max_frames):
            # Get frames (repeat last if episode ended)
            frame1 = all_frames[0][min(i, len(all_frames[0]) - 1)]
            frame2 = all_frames[1][min(i, len(all_frames[1]) - 1)]

            # Combine side by side
            img1 = Image.fromarray(frame1)
            img2 = Image.fromarray(frame2)

            combined_width = img1.width + img2.width
            combined_height = max(img1.height, img2.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))

            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (img1.width, 0))

            combined_frames.append(np.array(combined_img))

        # Save comparison
        filename = f"{RECORDINGS_DIR}/comparison.gif"
        imageio.mimsave(
            filename, combined_frames, duration=VIDEO_DURATION * 1.5, loop=0
        )
        print(f"💾 Saved comparison: {filename}")


def create_best_episode_video():
    """Record until we get a really good episode (fast completion)"""

    model = load_model_safe()
    if model is None:
        print("❌ No trained model found!")
        return

    create_directories()
    renderer = SimpleRenderer(GRID_SIZE)

    print("🎯 Recording best performance episode...")

    best_frames = None
    best_steps = float("inf")
    best_reward = -float("inf")

    # Try multiple episodes to find a good one
    max_attempts = 15
    target_steps = min(20, MAX_STEPS // 3)  # Target: complete in under 1/3 of max steps

    for attempt in range(max_attempts):
        env = make_env()
        obs, _ = env.reset()

        frames = []
        frame = renderer.render(env.get_obs())
        frames.append(frame)

        episode_reward = 0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            frame = renderer.render(env.get_obs())
            frames.append(frame)

            if terminated or truncated:
                if terminated and step + 1 < best_steps:
                    # Found a better episode!
                    best_steps = step + 1
                    best_reward = episode_reward
                    best_frames = frames.copy()
                    print(
                        f"  🏆 New best: Episode completed in {step + 1} steps (reward: {episode_reward:.1f})"
                    )

                    # If we found a really good episode, stop searching
                    if step + 1 <= target_steps:
                        print(f"  ✨ Excellent performance achieved! Stopping search.")
                        break
                else:
                    print(
                        f"  📊 Attempt {attempt + 1}: {step + 1} steps ({'completed' if terminated else 'timeout'})"
                    )
                break

    # Save the best episode found
    if best_frames:
        filename = f"{RECORDINGS_DIR}/best_episode_{best_steps}_steps.{VIDEO_FORMAT}"

        if VIDEO_FORMAT.lower() == "gif":
            imageio.mimsave(filename, best_frames, duration=VIDEO_DURATION, loop=0)
        elif VIDEO_FORMAT.lower() == "mp4":
            try:
                imageio.mimsave(filename, best_frames, fps=VIDEO_FPS)
            except Exception as e:
                print(f"MP4 save failed: {e}, saving as GIF instead")
                gif_filename = f"{RECORDINGS_DIR}/best_episode_{best_steps}_steps.gif"
                imageio.mimsave(
                    gif_filename, best_frames, duration=VIDEO_DURATION, loop=0
                )
                filename = gif_filename

        print(f"🏆 Saved best episode: {filename}")
        print(f"   Performance: {best_steps} steps, {best_reward:.1f} reward")
    else:
        print("⚠️  No completed episodes found in {max_attempts} attempts")


if __name__ == "__main__":
    print("🎬 Agent Video Recorder")
    print("=" * 30)
    print_config()

    choice = input(
        "Choose: [1] Record episodes, [2] Comparison video, [3] Best episode, [4] All: "
    ).strip()

    if choice == "1":
        create_agent_video()
    elif choice == "2":
        create_comparison_video()
    elif choice == "3":
        create_best_episode_video()
    else:  # Default to all
        create_agent_video()
        create_comparison_video()
        create_best_episode_video()
