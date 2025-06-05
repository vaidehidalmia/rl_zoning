import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class ZoningRenderer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)
        self.ax.set_aspect("equal")

        # Set up grid
        self.ax.set_xticks(range(grid_size))
        self.ax.set_yticks(range(grid_size))
        self.ax.grid(True, alpha=0.3)

        # Colors
        self.zone_colors = {1: "lightcoral", 2: "lightblue"}
        self.object_colors = {1: "red", 2: "blue"}

    def render(self, obs, return_array=True):
        self.ax.clear()

        # Set up the plot again
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, alpha=0.3)

        # Draw zones (background)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                zone_type = obs[r, c, 1]
                if zone_type in self.zone_colors:
                    self.ax.add_patch(
                        plt.Rectangle(
                            (c - 0.5, r - 0.5),
                            1,
                            1,
                            facecolor=self.zone_colors[zone_type],
                            alpha=0.3,
                            edgecolor="none",
                        )
                    )

        # Draw objects
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                obj_type = obs[r, c, 0]
                if obj_type > 0:
                    color = self.object_colors.get(obj_type, "gray")
                    self.ax.add_patch(
                        plt.Circle(
                            (c, r), 0.3, facecolor=color, edgecolor="black", linewidth=2
                        )
                    )

        # Draw agent
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                agent_info = obs[r, c, 2]
                if agent_info > 0:
                    if agent_info == 1:  # Agent not carrying
                        marker = "X"
                        color = "black"
                        size = 200
                    else:  # Agent carrying object (encoded as obj_type + 10)
                        marker = "X"
                        carried_type = agent_info - 10
                        color = self.object_colors.get(carried_type, "gray")
                        size = 300

                    self.ax.scatter(
                        c,
                        r,
                        marker=marker,
                        c=color,
                        s=size,
                        edgecolors="white",
                        linewidths=2,
                        zorder=10,
                    )

        # Add title and labels
        self.ax.set_title("Zoning Environment", fontsize=16, fontweight="bold")
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")

        # Invert y-axis to match array indexing
        self.ax.invert_yaxis()

        if return_array:
            # FIXED: Use the modern matplotlib API
            self.fig.canvas.draw()

            # Try modern API first, fall back to older versions
            try:
                # Modern matplotlib (3.5+)
                buf = self.fig.canvas.buffer_rgba()
                buf = np.asarray(buf)
                # Convert RGBA to RGB
                buf = buf[:, :, :3]
            except AttributeError:
                try:
                    # Older matplotlib (3.0-3.4)
                    buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                except AttributeError:
                    try:
                        # Even older matplotlib
                        buf = np.frombuffer(
                            self.fig.canvas.tostring_argb(), dtype=np.uint8
                        )
                        buf = buf.reshape(
                            self.fig.canvas.get_width_height()[::-1] + (4,)
                        )
                        # Convert ARGB to RGB
                        buf = buf[:, :, 1:4]
                    except AttributeError:
                        # Last resort - create a simple array
                        width, height = self.fig.canvas.get_width_height()
                        buf = np.ones((height, width, 3), dtype=np.uint8) * 128
                        print("Warning: Using fallback rendering array")

            return buf
        else:
            plt.show()
            return None

    def close(self):
        plt.close(self.fig)


# Test function to verify the renderer works
def test_renderer():
    """Test the fixed renderer"""
    print("🧪 Testing fixed renderer...")

    try:
        renderer = ZoningRenderer(4)

        # Create test observation
        obs = np.zeros((4, 4, 3), dtype=np.int32)

        # Add zones
        obs[:, :2, 1] = 1  # Left side = red zone
        obs[:, 2:, 1] = 2  # Right side = blue zone

        # Add an object
        obs[1, 3, 0] = 1  # Red object in blue zone (wrong)

        # Add agent
        obs[0, 0, 2] = 1  # Agent at top-left

        # Test rendering
        frame = renderer.render(obs, return_array=True)

        if frame is not None and isinstance(frame, np.ndarray):
            print(f"✅ Renderer works! Frame shape: {frame.shape}")
            renderer.close()
            return True
        else:
            print("❌ Renderer returned invalid data")
            renderer.close()
            return False

    except Exception as e:
        print(f"❌ Renderer test failed: {e}")
        return False


if __name__ == "__main__":
    test_renderer()
