# NOT USED IN THIS VERSION


class RewardConfig:
    # === PICKUP REWARDS ===
    PICKUP_GOOD_OBJECT = 10.0  # Reduced pickup reward
    PICKUP_CORRECT_OBJECT = -25.0  # Don't disturb correct objects
    PICKUP_INVALID = -10.0  # Invalid pickup penalty

    # === DROP REWARDS ===
    DROP_CORRECT_ZONE = 50.0  # Main goal - BIG reward!
    DROP_WRONG_ZONE = -5.0  # Penalty for wrong zone
    DROP_INVALID = -5.0  # Invalid drop penalty

    # === MOVEMENT REWARDS ===
    MOVEMENT_INVALID = -1.0  # Wall bump penalty

    # === PROGRESS REWARDS (distance-based shaping) ===
    DISTANCE_IMPROVEMENT_SCALE = 1.0  # Reward for getting closer to target
    CARRYING_BONUS = 0.5  # Small bonus per step while carrying

    # === EPISODE REWARDS ===
    STEP_PENALTY = -0.02  # Very small step penalty
    COMPLETION_BONUS = 100.0  # Big completion reward

    # === REMOVE DISCOVERY SYSTEM ENTIRELY ===
    DISCOVERY_BONUS = 0.0  # DISABLED - no discovery rewards

    # === ANTI-EXPLOITATION ===
    OSCILLATION_PENALTY = -5.0  # Strong oscillation penalty
    LOOP_PENALTY = -3.0  # Position loop penalty
