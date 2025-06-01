import gymnasium as gym
import numpy as np
from visualize.renderer import ZoningRenderer
from rewards import RewardConfig

class ZoningEnv(gym.Env):
    """
    Reinforcement Learning Environment for Object Placement in Zones

    The agent's goal is to place objects in their correct zones while avoiding
    disturbing objects that are already correctly placed.
    """

    def __init__(self, grid_size=4, num_objects=2, render_mode=None):
        """
        Initialize the Zoning Environment

        Args:
            grid_size (int): Size of the square grid
            num_objects (int): Number of objects to place
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.rewards = RewardConfig()

        # Episode management
        self.max_steps = self.grid_size * self.grid_size * 10
        self.current_step = 0

        # Object types
        self.object_types = {
            1: "red",
            2: "blue",
        }

        # Zone definitions with spatial areas and allowed object types
        self.zone_types = {
            1: {
                "name": "red_zone",
                "allowed_objects": [1],
                "area": lambda r, c: c < self.grid_size // 2  # Left half
            },
            2: {
                "name": "blue_zone",
                "allowed_objects": [2],
                "area": lambda r, c: c >= self.grid_size // 2  # Right half
            }
        }

        # Agent state
        self.agent_pos = (0, 0)
        self.carried_object = -1  # -1 means not carrying anything
        self.objects = {}  # Dictionary mapping positions to object types

        self.render_mode = render_mode
        self.renderer = ZoningRenderer(grid_size) if render_mode else None

        # Termination flags
        self.terminated = False
        self.truncated = False

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Pickup, 5=Drop
        self.action_space = gym.spaces.Discrete(6)

        # Observation: 3-channel grid (objects, zones, agent+carried)
        self.observation_space = gym.spaces.Box(
            low=0, high=10 + len(self.object_types),
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused but required by Gymnasium)

        Returns:
            observation: Initial state observation
            info: Empty dict (Gymnasium standard)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.terminated = False
        self.truncated = False

        # Random agent position
        self.agent_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )
        self.carried_object = -1
        self.objects.clear()

        # Get free positions
        free_positions = [(r, c) for r in range(self.grid_size) 
                        for c in range(self.grid_size) 
                        if (r, c) != self.agent_pos]
        
        # Shuffle for random placement
        np.random.shuffle(free_positions)
        
        # Place objects in wrong zones
        for i in range(min(self.num_objects, len(free_positions))):
            pos = free_positions[i]
            r, c = pos
            
            # Simple rule: left half gets blue (wrong), right half gets red (wrong)
            obj_type = 2 if c < self.grid_size // 2 else 1
            self.objects[pos] = obj_type

        return self.get_obs(), {}

    def step(self, action):
        """Enhanced step with object-seeking rewards"""
        self.current_step += 1
        reward = 0
        
        # Store previous position for distance calculations
        prev_pos = self.agent_pos
        r, c = self.agent_pos
        
        # Compute distances before action
        prev_distance = self.get_distance_to_target_zone()
        prev_object_distance = self.get_distance_to_nearest_object()

        # Movement actions (0-3) - SAME AS BEFORE
        if action == 0 and r > 0:  # Up
            self.agent_pos = (r - 1, c)
        elif action == 1 and r < self.grid_size - 1:  # Down
            self.agent_pos = (r + 1, c)
        elif action == 2 and c > 0:  # Left
            self.agent_pos = (r, c - 1)
        elif action == 3 and c < self.grid_size - 1:  # Right
            self.agent_pos = (r, c + 1)

        # Pickup action (4) - SAME AS BEFORE
        elif action == 4:
            if self.carried_object == -1 and self.agent_pos in self.objects:
                reward += 10.0
                obj_at_pos = self.objects[self.agent_pos]
                if self.is_correct_zone(self.agent_pos, obj_at_pos):
                    reward = self.rewards.PICKUP_CORRECT_OBJECT
                    self.carried_object = self.objects.pop(self.agent_pos)
                else:
                    self.carried_object = self.objects.pop(self.agent_pos)
                    reward = self.rewards.PICKUP_GOOD_OBJECT
            else:
                reward = self.rewards.PICKUP_INVALID

        # Drop action (5) - SAME AS BEFORE
        elif action == 5:
            if self.carried_object != -1:
                if self.agent_pos not in self.objects:
                    self.objects[self.agent_pos] = self.carried_object
                    if self.is_correct_zone(self.agent_pos, self.carried_object):
                        reward = self.rewards.DROP_CORRECT_ZONE
                    else:
                        reward = self.rewards.DROP_WRONG_ZONE
                    self.carried_object = -1
                else:
                    reward = self.rewards.DROP_INVALID
            else:
                reward = self.rewards.DROP_INVALID
        else:
            reward = self.rewards.MOVEMENT_INVALID

        # === NEW: OBJECT-SEEKING REWARD ===
        if self.carried_object == -1 and self.objects:  # Not carrying, objects exist
            new_object_distance = self.get_distance_to_nearest_object()
            if prev_object_distance is not None and new_object_distance is not None:
                if new_object_distance < prev_object_distance:  # Got closer to an object
                    reward += 2
                elif new_object_distance > prev_object_distance:  # Got farther
                    reward -= 2

        # # Existing distance shaping (for when carrying object)
        new_distance = self.get_distance_to_target_zone()
        if prev_distance is not None and new_distance is not None:
            distance_delta = prev_distance - new_distance
            reward += max(0.0, distance_delta * self.rewards.DISTANCE_SHAPING_SCALE)

        # Step penalty and completion
        reward += self.rewards.STEP_PENALTY
        self.terminated = self.is_episode_complete()
        self.truncated = self.current_step >= self.max_steps

        if self.terminated and not self.truncated:
            reward += self.rewards.COMPLETION_BONUS

        return self.get_obs(), reward, self.terminated, self.truncated, {}


    # Add this helper method to your ZoningEnv class:
    def get_distance_to_nearest_object(self):
        """Calculate Manhattan distance to nearest object"""
        if not self.objects:
            return None
            
        min_dist = float("inf")
        ar, ac = self.agent_pos
        
        for (r, c) in self.objects.keys():
            dist = abs(r - ar) + abs(c - ac)
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float("inf") else None

    def is_episode_complete(self):
        """
        Check if all objects are correctly placed and the agent is not carrying anything
        """
        return self.carried_object == -1 and all(
            self.is_correct_zone(pos, obj) for pos, obj in self.objects.items()
        )

    def is_correct_zone(self, pos, obj_type):
        """
        Check if an object is in its correct zone

        Args:
            pos (tuple): Position (row, col)
            obj_type (int): Object type identifier

        Returns:
            bool: True if object is in correct zone
        """
        return any(
            obj_type in zone["allowed_objects"] and zone["area"](*pos)
            for zone in self.zone_types.values()
        )

    def get_distance_to_target_zone(self):
        """
        Calculate Manhattan distance to nearest valid drop zone for the carried object.
        Returns None if not carrying anything.
        """
        if self.carried_object == -1:
            return None

        min_dist = float("inf")
        ar, ac = self.agent_pos

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.is_correct_zone((r, c), self.carried_object):
                    dist = abs(r - ar) + abs(c - ac)
                    min_dist = min(min_dist, dist)

        return min_dist if min_dist != float("inf") else None

    def get_obs(self):
        """
        Generate observation of current state - modified for CNN compatibility

        Returns:
            np.ndarray: 3-channel observation array
                Channel 0: Objects (1-3 for object types, 0 for empty)
                Channel 1: Zones (1-3 for zone types, 0 for no zone)
                Channel 2: Agent and carried object info
        """
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.int32)

        # Channel 0: Object positions and types
        for (r, c), obj_type in self.objects.items():
            obs[r, c, 0] = obj_type

        # Channel 1: Zone definitions
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for zone_id, zone in self.zone_types.items():
                    if zone["area"](r, c):
                        obs[r, c, 1] = zone_id
                        break

        # Channel 2: Agent position and carried object
        ar, ac = self.agent_pos
        obs[ar, ac, 2] = 1 if self.carried_object == -1 else self.carried_object + 10

        return obs

    
    def render(self):
        """
        Render using matplotlib in 'human' or 'rgb_array' mode.
        """
        if self.renderer:
            obs = self.get_obs()
            if self.render_mode == "human":
                self.renderer.render(obs, return_array=False)
            elif self.render_mode == "rgb_array":
                return self.renderer.render(obs, return_array=True)
        else:
            raise NotImplementedError("Render mode not enabled.")