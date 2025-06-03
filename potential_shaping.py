class PotentialShaping:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.previous_potential = 0

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_target_zone_center(self, obj_type, grid_size):
        """Get the center position of the target zone for an object type"""
        if obj_type == 1:  # Red objects go to left zone
            return (grid_size // 2, grid_size // 4)
        elif obj_type == 2:  # Blue objects go to right zone
            return (grid_size // 2, 3 * grid_size // 4)
        return (grid_size // 2, grid_size // 2)  # Default center

    def is_correct_zone(self, pos, obj_type, grid_size):
        """Check if position is in correct zone for object type"""
        r, c = pos
        if obj_type == 1:  # Red zone (left)
            return c < grid_size // 2
        elif obj_type == 2:  # Blue zone (right)
            return c >= grid_size // 2
        return False

    def get_potential(self, agent_pos, carried_object, objects, grid_size):
        """
        Calculate potential based on current state:
        - If carrying object: potential based on distance to correct zone
        - If not carrying: potential based on distance to nearest misplaced object
        """
        if carried_object != -1:
            # Agent is carrying an object - guide to correct zone
            target_zone_center = self.get_target_zone_center(carried_object, grid_size)
            distance_to_zone = self.manhattan_distance(agent_pos, target_zone_center)
            return -distance_to_zone

        else:
            # Agent not carrying - guide to nearest misplaced object
            misplaced_objects = []

            for pos, obj_type in objects.items():
                if not self.is_correct_zone(pos, obj_type, grid_size):
                    misplaced_objects.append(pos)

            if not misplaced_objects:
                return 0  # No misplaced objects, no guidance needed

            # Find distance to nearest misplaced object
            min_distance = min(
                self.manhattan_distance(agent_pos, obj_pos)
                for obj_pos in misplaced_objects
            )
            return -min_distance

    def get_shaping_reward(self, agent_pos, carried_object, objects, grid_size):
        """
        Calculate the potential-based shaping reward
        """
        current_potential = self.get_potential(
            agent_pos, carried_object, objects, grid_size
        )

        # Shaping reward = γ * Φ(s') - Φ(s)
        shaping_reward = self.gamma * current_potential - self.previous_potential

        # Update for next step
        self.previous_potential = current_potential

        return shaping_reward

    def reset(self, agent_pos, carried_object, objects, grid_size):
        """Reset shaping for new episode"""
        self.previous_potential = self.get_potential(
            agent_pos, carried_object, objects, grid_size
        )
