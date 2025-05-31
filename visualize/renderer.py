import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class ZoningRenderer:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def render(self, obs, return_array=False):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title("Zoning Environment")

        obj_layer = obs[:, :, 0]
        zone_layer = obs[:, :, 1]
        agent_layer = obs[:, :, 2]

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Zones: colored backgrounds
                if zone_layer[r, c] == 1:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, color="mistyrose", zorder=0))
                elif zone_layer[r, c] == 2:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, color="lightblue", zorder=0))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                label = ""
                color = "black"

                # Object only
                if obj_layer[r, c] == 1:
                    label = "R"  # Red object
                    color = "red"
                elif obj_layer[r, c] == 2:
                    label = "B"  # Blue object
                    color = "blue"

                # Agent only
                if agent_layer[r, c] == 1:
                    label = "A"  # Agent alone
                    color = "black"

                # Agent carrying object
                elif agent_layer[r, c] >= 11:
                    carried_type = agent_layer[r, c] - 10
                    if carried_type == 1:
                        label = "AR"  # Agent + red
                        color = "darkred"
                    elif carried_type == 2:
                        label = "AB"  # Agent + blue
                        color = "darkblue"
                    else:
                        label = "A?"
                        color = "gray"

                # Draw label
                if label:
                    ax.text(c + 0.5, r + 0.5, label, ha='center', va='center',
                            fontsize=12, fontweight='bold', color=color)

        plt.tight_layout()
        canvas = FigureCanvas(fig)

        if return_array:
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img

        plt.show()