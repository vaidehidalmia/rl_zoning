import gymnasium as gym
import numpy as np
from visualize.renderer import ZoningRenderer

class ZoningEnv(gym.Env):
    """
    Reinforcement Learning Environment for Object Placement in Zones

    The agent's goal is to place objects in their correct zones while avoiding
    disturbing objects that are already correctly placed.
    """

    def __init__(self, grid_size=6, num_objects=2, render_mode=None):
        """
        Initialize the Zoning Environment

        Args:
            grid_size (int): Size of the square grid
            num_objects (int): Number of objects to place
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_objects = num_objects

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

        # Random agent starting position
        self.agent_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )
        self.carried_object = -1
        self.objects.clear()

        # Place objects randomly, ensuring no overlaps
        all_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - {self.agent_pos})
        np.random.shuffle(free_positions)
        num_to_place = min(self.num_objects, len(free_positions))

        for i in range(num_to_place):
            pos = free_positions[i]
            obj_type = np.random.choice(list(self.object_types.keys()))
            self.objects[pos] = obj_type

        return self.get_obs(), {}

    def step(self, action):
        """
        Execute one environment step

        Args:
            action (int): Action to take (0-5)

        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information (empty)
        """
        self.current_step += 1
        reward = 0

        # Get current position
        r, c = self.agent_pos

        # Compute distance before taking action (for reward shaping)
        prev_distance = self.get_distance_to_target_zone()

        # Movement actions (0-3)
        if action == 0 and r > 0:  # Up
            self.agent_pos = (r - 1, c)
        elif action == 1 and r < self.grid_size - 1:  # Down
            self.agent_pos = (r + 1, c)
        elif action == 2 and c > 0:  # Left
            self.agent_pos = (r, c - 1)
        elif action == 3 and c < self.grid_size - 1:  # Right
            self.agent_pos = (r, c + 1)

        # Pickup action (4)
        elif action == 4:
            if self.carried_object == -1 and self.agent_pos in self.objects:
                obj_at_pos = self.objects[self.agent_pos]

                # Check if object is already correctly placed
                if self.is_correct_zone(self.agent_pos, obj_at_pos):
                    # Penalty for disturbing correctly placed objects
                    reward = -5
                    self.carried_object = self.objects.pop(self.agent_pos)
                else:
                    # Good reward for picking up misplaced objects
                    self.carried_object = self.objects.pop(self.agent_pos)
                    reward = 3
            else:
                # Penalty for invalid pickup
                reward = -0.1

        # Drop action (5)
        elif action == 5:
            if self.carried_object != -1:
                # Check if position is free
                if self.agent_pos not in self.objects:
                    # Place the object
                    self.objects[self.agent_pos] = self.carried_object

                    # Large reward/penalty based on correctness
                    if self.is_correct_zone(self.agent_pos, self.carried_object):
                        reward = 5  # Large reward for correct placement
                    else:
                        reward = -3  # Penalty for incorrect placement

                    self.carried_object = -1
                else:
                    # Penalty for trying to drop on occupied position
                    reward = -1
            else:
                # Penalty for trying to drop when not carrying
                reward = -1

        # Invalid action (e.g. wall bump)
        else:
            reward = -1.0

        # Compute distance after action
        new_distance = self.get_distance_to_target_zone()

        # Positive shaping reward for moving closer to target zone
        if prev_distance is not None and new_distance is not None:
            distance_delta = prev_distance - new_distance
            reward += max(0.0, distance_delta * 0.5)

        # Step penalty to encourage efficiency
        reward -= 0.005

        # Check termination and truncation conditions
        self.terminated = self.is_episode_complete()
        self.truncated = self.current_step >= self.max_steps

        # Big reward bonus for completing the task
        if self.terminated and not self.truncated:
            reward += 100

        return self.get_obs(), reward, self.terminated, self.truncated, {}

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