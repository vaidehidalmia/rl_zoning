import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

class ZoningRenderer:
    """Renderer for the zoning environment - FIXED for video recording"""

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.fig = None
        self.ax = None
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Flip Y axis to match array indexing
        
        # Remove axes for cleaner look
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, alpha=0.3)
        
        # Set title
        self.ax.set_title("Agent Zoning Task", fontsize=16, fontweight='bold')

    def render(self, obs, return_array=False):
        """Render the current state"""
        # Clear previous frame
        self.ax.clear()
        self.setup_plot()
        
        # Draw zone backgrounds
        self._draw_zones()
        
        # Draw grid
        self._draw_grid()
        
        # Draw objects and agent from observation
        self._draw_from_observation(obs)
        
        if return_array:
            # FIXED: Use modern matplotlib method
            self.fig.canvas.draw()
            
            # Get the RGBA buffer and convert to RGB
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            return buf
        else:
            plt.pause(0.1)
            return None

    def _draw_zones(self):
        """Draw colored zone backgrounds"""
        # Red zone (left side)
        red_zone = patches.Rectangle(
            (-0.5, -0.5), self.grid_size // 2, self.grid_size,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.1
        )
        self.ax.add_patch(red_zone)
        
        # Blue zone (right side)
        blue_zone = patches.Rectangle(
            (self.grid_size // 2 - 0.5, -0.5), self.grid_size // 2, self.grid_size,
            linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1
        )
        self.ax.add_patch(blue_zone)
        
        # Add zone labels
        self.ax.text(self.grid_size // 4 - 0.5, -0.8, 'RED ZONE', 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='red')
        self.ax.text(3 * self.grid_size // 4 - 0.5, -0.8, 'BLUE ZONE', 
                    ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

    def _draw_grid(self):
        """Draw grid lines"""
        for i in range(self.grid_size + 1):
            self.ax.axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
            self.ax.axvline(i - 0.5, color='black', linewidth=1, alpha=0.3)

    def _draw_from_observation(self, obs):
        """Draw objects and agent from observation array"""
        # obs shape: (grid_size, grid_size, 3)
        # Channel 0: Object positions and types
        # Channel 1: Zone definitions  
        # Channel 2: Agent position and carried object

        # Draw objects
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                obj_type = obs[r, c, 0]
                if obj_type > 0:
                    self._draw_object(r, c, obj_type)

        # Draw agent
        agent_pos = None
        carried_object = -1
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                agent_indicator = obs[r, c, 2]
                if agent_indicator > 0:
                    agent_pos = (r, c)
                    if agent_indicator == 1:
                        carried_object = -1  # Not carrying
                    else:
                        carried_object = agent_indicator - 10  # Decode carried object
                    break
        
        if agent_pos:
            self._draw_agent(agent_pos[0], agent_pos[1], carried_object)

    def _draw_object(self, row, col, obj_type):
        """Draw an object at the specified position"""
        colors = {1: 'red', 2: 'blue'}
        color = colors.get(obj_type, 'gray')
        
        # Draw object as a square
        obj_patch = patches.Rectangle(
            (col - 0.3, row - 0.3), 0.6, 0.6,
            linewidth=2, edgecolor='black', facecolor=color, alpha=0.8
        )
        self.ax.add_patch(obj_patch)
        
        # Add object type label
        self.ax.text(col, row, str(obj_type), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')

    def _draw_agent(self, row, col, carried_object=-1):
        """Draw the agent at the specified position"""
        # Draw agent as a circle
        agent_circle = patches.Circle(
            (col, row), 0.25, linewidth=3, edgecolor='black', 
            facecolor='yellow', alpha=0.9
        )
        self.ax.add_patch(agent_circle)
        
        # Add agent symbol
        self.ax.text(col, row, '🤖', ha='center', va='center', fontsize=16)
        
        # Show carried object
        if carried_object > 0:
            colors = {1: 'red', 2: 'blue'}
            color = colors.get(carried_object, 'gray')
            
            # Draw small carried object indicator
            carried_patch = patches.Circle(
                (col + 0.15, row - 0.15), 0.1, 
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.9
            )
            self.ax.add_patch(carried_patch)
            
            # Add carrying text
            self.ax.text(col, row + 0.45, f'Carrying: {carried_object}', 
                        ha='center', va='center', fontsize=10, fontweight='bold')

    def close(self):
        """Close the renderer"""
        if self.fig:
            plt.close(self.fig)