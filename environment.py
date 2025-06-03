import gymnasium as gym
import numpy as np
from visualize.renderer import ZoningRenderer
from potential_shaping import PotentialShaping

class RobustRewards:
    
    # ONLY positive reward: completing the task
    TASK_COMPLETE = 100.0
    
    # Small negative rewards for efficiency
    STEP_COST = -0.1           # Encourage efficiency
    INVALID_ACTION = -1.0      # Discourage invalid actions



class ZoningEnv(gym.Env):

    def __init__(self, grid_size=4, num_objects=1, render_mode=None, use_shaping=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.rewards = RobustRewards()
        self.use_shaping = use_shaping

        # Episode management  
        self.max_steps = 200
        self.current_step = 0

        # Object types and zones
        self.zone_types = {
            1: lambda r, c: c < self.grid_size // 2,    # Red zone (left)
            2: lambda r, c: c >= self.grid_size // 2,   # Blue zone (right)
        }

        # Agent state
        self.agent_pos = (0, 0)
        self.carried_object = -1
        self.objects = {}

        # Potential-based shaping
        if self.use_shaping:
            self.shaping = PotentialShaping()

        # Rendering
        self.render_mode = render_mode
        self.renderer = ZoningRenderer(grid_size) if render_mode else None

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0, high=12,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset agent state
        self.agent_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.carried_object = -1
        self.objects.clear()

        # Place objects in wrong zones
        free_positions = [(r, c) for r in range(self.grid_size) 
                         for c in range(self.grid_size) 
                         if (r, c) != self.agent_pos]
        np.random.shuffle(free_positions)
        
        for i in range(min(self.num_objects, len(free_positions))):
            pos = free_positions[i]
            r, c = pos
            obj_type = 2 if c < self.grid_size // 2 else 1  # Wrong zone
            self.objects[pos] = obj_type

        # Reset shaping
        if self.use_shaping:
            self.shaping.reset(self.agent_pos, self.carried_object, self.objects, self.grid_size)

        return self.get_obs(), {}

    def step(self, action):
        self.current_step += 1
        reward = self.rewards.STEP_COST 
        
        r, c = self.agent_pos

        # Movement actions (0-3) 
        if action == 0 and r > 0:
            self.agent_pos = (r - 1, c)
        elif action == 1 and r < self.grid_size - 1:
            self.agent_pos = (r + 1, c)
        elif action == 2 and c > 0:
            self.agent_pos = (r, c - 1)
        elif action == 3 and c < self.grid_size - 1:
            self.agent_pos = (r, c + 1)

        # Pickup action (4)
        elif action == 4:
            if self.carried_object == -1 and self.agent_pos in self.objects:
                self.carried_object = self.objects.pop(self.agent_pos)
            else:
                reward += self.rewards.INVALID_ACTION

        # Drop action (5) 
        elif action == 5:
            if self.carried_object != -1 and self.agent_pos not in self.objects:
                self.objects[self.agent_pos] = self.carried_object
                self.carried_object = -1
            else:
                reward += self.rewards.INVALID_ACTION
        else:
            reward += self.rewards.INVALID_ACTION

        # Add potential-based shaping reward
        if self.use_shaping:
            shaping_reward = self.shaping.get_shaping_reward(
                self.agent_pos, self.carried_object, self.objects, self.grid_size
            )
            reward += shaping_reward

        # Check completion
        terminated = self.is_episode_complete()
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += self.rewards.TASK_COMPLETE
            print(f"🎯 TASK COMPLETED! Reward: +{self.rewards.TASK_COMPLETE}")

        return self.get_obs(), reward, terminated, truncated, {}

    def is_correct_zone(self, pos, obj_type):
        zone_func = self.zone_types.get(obj_type)
        return zone_func(*pos) if zone_func else False

    def is_episode_complete(self):
        """Task complete when all objects correctly placed AND not carrying"""
        if self.carried_object != -1:
            return False
            
        if not self.objects:  # No objects left
            return False
            
        return all(self.is_correct_zone(pos, obj) for pos, obj in self.objects.items())

    def get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.int32)

        # Channel 0: Object positions and types
        for (r, c), obj_type in self.objects.items():
            obs[r, c, 0] = obj_type

        # Channel 1: Zone definitions
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                obs[r, c, 1] = 1 if c < self.grid_size // 2 else 2

        # Channel 2: Agent position and carried object
        ar, ac = self.agent_pos
        obs[ar, ac, 2] = 1 if self.carried_object == -1 else self.carried_object + 10

        return obs

    def render(self):
        if self.renderer:
            obs = self.get_obs()
            if self.render_mode == "human":
                self.renderer.render(obs, return_array=False)
            elif self.render_mode == "rgb_array":
                return self.renderer.render(obs, return_array=True)
        else:
            raise NotImplementedError("Render mode not enabled.")