import os
import numpy as np
from PIL import Image
import imageio
from stable_baselines3 import PPO
from environment import ZoningEnv
from config import GRID_SIZE, NUM_OBJECTS, MODEL_PATH

# Fix matplotlib backend for video recording
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# FIXED RENDERER - replace the broken one
class VideoZoningRenderer:
    """Simple renderer that works for video recording"""
    
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        
    def render(self, obs, return_array=True):
        """Create a simple visualization"""        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5) 
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title("Agent Performance", fontsize=14, fontweight='bold')
        
        # Draw zones
        # Red zone (left)
        red_zone = patches.Rectangle(
            (-0.5, -0.5), self.grid_size // 2, self.grid_size,
            facecolor='red', alpha=0.2, edgecolor='red', linewidth=2
        )
        ax.add_patch(red_zone)
        
        # Blue zone (right)  
        blue_zone = patches.Rectangle(
            (self.grid_size // 2 - 0.5, -0.5), self.grid_size // 2, self.grid_size,
            facecolor='blue', alpha=0.2, edgecolor='blue', linewidth=2
        )
        ax.add_patch(blue_zone)
        
        # Draw from observation
        agent_pos = None
        carried_obj = -1
        
        # Find objects and agent
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Objects (channel 0)
                obj_type = obs[r, c, 0]
                if obj_type > 0:
                    color = 'red' if obj_type == 1 else 'blue'
                    obj_rect = patches.Rectangle(
                        (c - 0.3, r - 0.3), 0.6, 0.6,
                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.8
                    )
                    ax.add_patch(obj_rect)
                    ax.text(c, r, str(obj_type), ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='white')
                
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
                (c, r), 0.25, facecolor='yellow', edgecolor='black', linewidth=3
            )
            ax.add_patch(agent_circle)
            ax.text(c, r, 'A', ha='center', va='center', fontsize=12, fontweight='bold')
            
            if carried_obj > 0:
                ax.text(c, r + 0.4, f'Carrying: {carried_obj}', ha='center', va='center', 
                       fontsize=8, fontweight='bold')
        
        # Convert to array
        if return_array:
            # FIXED: Use buffer method that works on all platforms
            fig.canvas.draw()
            
            # Get canvas as RGBA buffer
            width, height = fig.canvas.get_width_height()
            
            # Try multiple methods for cross-platform compatibility
            try:
                # Method 1: Modern approach
                buf = fig.canvas.buffer_rgba()
                buf = np.asarray(buf)
                buf = buf.reshape(height, width, 4)  # RGBA
                buf = buf[:, :, :3]  # Convert to RGB
            except:
                try:
                    # Method 2: Alternative buffer method
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(height, width, 3)
                except:
                    # Method 3: Fallback - save to temporary array
                    import io
                    buf_io = io.BytesIO()
                    fig.savefig(buf_io, format='png', bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    buf_io.seek(0)
                    from PIL import Image
                    img = Image.open(buf_io)
                    buf = np.array(img)
                    if buf.shape[2] == 4:  # RGBA to RGB
                        buf = buf[:, :, :3]
            
            plt.close(fig)  # Important: close figure to free memory
            return buf
        else:
            plt.show()
            plt.close(fig)
            return None

def find_model_file():
    """Find the correct model file"""
    # Try the exact same path that works in evaluate.py
    possible_paths = [
        "models/zoning_agent",  # This is what works in evaluate.py
        "models/zoning_agent.zip",
        MODEL_PATH,
        MODEL_PATH + ".zip",
        "zoning_agent",
        "zoning_agent.zip"
    ]
    
    print("🔍 Searching for model file...")
    for path in possible_paths:
        print(f"  Trying: {path}")
        if os.path.exists(path):
            print(f"  ✅ Found: {path}")
            return path
        elif os.path.exists(path + ".zip"):
            print(f"  ✅ Found: {path}.zip")
            return path + ".zip"
    
    # List available models for debugging
    print("\n📂 Debug info:")
    print(f"Current directory: {os.getcwd()}")
    
    if os.path.exists("models"):
        print("Available model files in models/:")
        for file in os.listdir("models"):
            print(f"  - models/{file}")
            # Try loading this file directly
            try:
                test_path = f"models/{file}"
                if not file.startswith('.'):  # Skip hidden files
                    print(f"    Testing if {test_path} is loadable...")
                    # Don't actually load, just check if it looks like a model
                    if os.path.isfile(test_path) or test_path.endswith('.zip'):
                        return test_path
            except:
                pass
    else:
        print("❌ models/ directory not found")
    
    print("Available files in current directory:")
    for file in os.listdir("."):
        if "agent" in file.lower() or file.endswith(".zip") or file == "models":
            print(f"  - {file}")
    
    return None

def create_video_of_agent(model_path, save_path="agent_performance", num_episodes=3, format="gif"):
    """
    Record agent performance as video/gif
    
    Args:
        save_path: Base name for saved files
        num_episodes: Number of episodes to record
        format: "gif" or "mp4"
    """
    
    # Find and load trained model
    print(f"📦 Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment with FIXED renderer
    env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode=None)
    env.renderer = VideoZoningRenderer(GRID_SIZE)  # Replace with working renderer
    
    for episode in range(num_episodes):
        print(f"🎬 Recording Episode {episode + 1}...")
        
        frames = []
        obs, _ = env.reset()
        
        # Add initial frame
        frame = env.renderer.render(env.get_obs(), return_array=True)
        if frame is not None:
            frames.append(frame)
        
        total_reward = 0
        
        for step in range(100):  # Max 100 steps
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Capture frame
            frame = env.renderer.render(env.get_obs(), return_array=True)
            if frame is not None:
                frames.append(frame)
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: {'✅ COMPLETED' if terminated else '❌ TIMEOUT'} in {step + 1} steps (reward: {total_reward:.1f})")
                break
        
        # Save episode as GIF or MP4
        if frames:
            filename = f"{save_path}_episode_{episode + 1}.{format}"
            
            if format.lower() == "gif":
                # Save as GIF
                imageio.mimsave(filename, frames, duration=0.5, loop=0)
                print(f"💾 Saved GIF: {filename}")
                
            elif format.lower() == "mp4":
                # Save as MP4 (requires ffmpeg)
                try:
                    imageio.mimsave(filename, frames, fps=2)
                    print(f"💾 Saved MP4: {filename}")
                except Exception as e:
                    print(f"❌ MP4 save failed: {e}")
                    # Fallback to GIF
                    gif_filename = f"{save_path}_episode_{episode + 1}.gif"
                    imageio.mimsave(gif_filename, frames, duration=0.5, loop=0)
                    print(f"💾 Saved as GIF instead: {gif_filename}")
        else:
            print(f"❌ No frames captured for episode {episode + 1}")
    
    print(f"🎯 Recording completed! Check for {save_path}_episode_*.{format} files")


def create_side_by_side_comparison(model_path):
    """Create a comparison showing multiple episodes side by side"""
    
    print("🎬 Creating side-by-side comparison...")
    model = PPO.load(model_path)
    env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode=None)
    env.renderer = VideoZoningRenderer(GRID_SIZE)  # Use fixed renderer
    
    all_episode_frames = []
    
    # Record 2 episodes
    for episode in range(2):
        frames = []
        obs, _ = env.reset()
        
        frame = env.renderer.render(env.get_obs(), return_array=True)
        if frame is not None:
            frames.append(frame)
        
        for step in range(50):  # Shorter episodes for comparison
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            frame = env.renderer.render(env.get_obs(), return_array=True)
            if frame is not None:
                frames.append(frame)
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: Completed in {step + 1} steps")
                break
        
        all_episode_frames.append(frames)
    
    # Create side-by-side frames
    if len(all_episode_frames) == 2:
        max_frames = max(len(all_episode_frames[0]), len(all_episode_frames[1]))
        combined_frames = []
        
        for i in range(max_frames):
            # Get frames (repeat last frame if episode ended)
            frame1 = all_episode_frames[0][min(i, len(all_episode_frames[0])-1)]
            frame2 = all_episode_frames[1][min(i, len(all_episode_frames[1])-1)]
            
            # Convert to PIL Images
            img1 = Image.fromarray(frame1)
            img2 = Image.fromarray(frame2)
            
            # Create side-by-side image
            combined_width = img1.width + img2.width
            combined_height = max(img1.height, img2.height)
            combined_img = Image.new('RGB', (combined_width, combined_height))
            
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (img1.width, 0))
            
            combined_frames.append(np.array(combined_img))
        
        # Save comparison GIF
        imageio.mimsave("agent_comparison.gif", combined_frames, duration=0.8, loop=0)
        print("💾 Saved comparison: agent_comparison.gif")


def create_best_episode_video(model_path):
    """Record until we get a really good episode (under 15 steps)"""
    
    print("🎯 Recording best performance episode...")
    model = PPO.load(model_path)
    env = ZoningEnv(grid_size=GRID_SIZE, num_objects=NUM_OBJECTS, render_mode=None)
    env.renderer = VideoZoningRenderer(GRID_SIZE)  # Use fixed renderer
    
    episode = 0
    while episode < 10:  # Try up to 10 episodes
        episode += 1
        frames = []
        obs, _ = env.reset()
        
        frame = env.renderer.render(env.get_obs(), return_array=True)
        if frame is not None:
            frames.append(frame)
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            frame = env.renderer.render(env.get_obs(), return_array=True)
            if frame is not None:
                frames.append(frame)
            
            if terminated or truncated:
                if terminated and step + 1 <= 20:  # Good performance
                    filename = f"best_episode_{step + 1}_steps.gif"
                    imageio.mimsave(filename, frames, duration=0.6, loop=0)
                    print(f"🏆 Recorded excellent episode: {filename} ({step + 1} steps)")
                    return
                else:
                    print(f"Episode {episode}: {step + 1} steps (looking for better...)")
                break
    
    print("⚠️  Didn't capture a really fast episode, but that's normal!")


if __name__ == "__main__":
    print("🎬 Agent Performance Video Recorder")
    print("=" * 50)
    
    # Test model loading first (same as evaluate.py)
    print("🧪 Testing model loading...")
    try:
        from config import MODEL_PATH
        print(f"Config MODEL_PATH: {MODEL_PATH}")
        
        # Try exact same loading as evaluate.py
        test_model = PPO.load(MODEL_PATH)
        print("✅ Model loads successfully with CONFIG path!")
        model_path = MODEL_PATH
    except Exception as e:
        print(f"❌ Config path failed: {e}")
        model_path = find_model_file()
        if model_path is None:
            print("❌ Could not find any model file!")
            exit(1)
        try:
            test_model = PPO.load(model_path)
            print(f"✅ Model loads successfully with: {model_path}")
        except Exception as e2:
            print(f"❌ Model loading failed: {e2}")
            exit(1)
    
    # Convert to absolute path before changing directories
    model_path = os.path.abspath(model_path)
    print(f"📦 Using absolute model path: {model_path}")
    
    # Create output directory
    os.makedirs("recordings", exist_ok=True)
    os.chdir("recordings")
    
    # Now proceed with video creation using the working model path
    print(f"\n📦 Model path from recordings directory: {model_path}")
    
    # Option 1: Regular episodes
    print("\n1️⃣ Recording regular episodes...")
    create_video_of_agent(model_path, "agent_performance", num_episodes=3, format="gif")
    
    # Option 2: Side-by-side comparison  
    print("\n2️⃣ Creating side-by-side comparison...")
    create_side_by_side_comparison(model_path)
    
    # Option 3: Best episode
    print("\n3️⃣ Recording best performance...")
    create_best_episode_video(model_path)
    
    print("\n🎯 All recordings completed!")
    print("📁 Check the 'recordings' folder for your videos!")
    print("\nGenerated files:")
    for file in os.listdir("."):
        if file.endswith((".gif", ".mp4")):
            print(f"  - {file}")