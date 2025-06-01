class RewardConfig:
    """Centralized reward configuration for easy tuning"""
    
    # # === PICKUP REWARDS ===
    # PICKUP_GOOD_OBJECT = 1.0          # Picking up misplaced object
    # PICKUP_CORRECT_OBJECT = -10.0     # Picking up correctly placed object (BAD!)
    # PICKUP_INVALID = -10.0            # Invalid pickup (no object, already carrying)
    
    # # === DROP REWARDS ===
    # DROP_CORRECT_ZONE = 5.0           # Dropping object in correct zone
    # DROP_WRONG_ZONE = -5.0            # Dropping object in wrong zone
    # DROP_INVALID = -10.0              # Invalid drop (not carrying, position occupied)
    
    # # === MOVEMENT REWARDS ===
    # MOVEMENT_INVALID = -1.0           # Wall bumps, invalid movement
    # DISTANCE_SHAPING_SCALE = 0.1      # Multiplier for distance-based reward shaping
    
    # # === EPISODE REWARDS ===
    # STEP_PENALTY = -0.01              # Small penalty per step (encourages efficiency)
    # COMPLETION_BONUS = 50.0           # Big reward for completing the task


    #  # === PICKUP REWARDS ===
    # PICKUP_GOOD_OBJECT = 20.0
    # PICKUP_CORRECT_OBJECT = -10.0
    # PICKUP_INVALID = -5.0
    
    # DROP_CORRECT_ZONE = 50.0           # BIG reward for any correct drop
    # DROP_WRONG_ZONE = 0.0              # No penalty - just learn to drop
    # DROP_INVALID = -5.0
    
    # DISTANCE_SHAPING_SCALE = 2.0       # Help with navigation
    # STEP_PENALTY = -0.001              # Almost no step penalty
    # COMPLETION_BONUS = 200.0

    # MOVEMENT_INVALID = 0
    
    # # === ADVANCED ANTI-EXPLOITATION ===
    # ENABLE_ANTI_LOOP = False
    # LOOP_PENALTY = -0.5
    # REPEAT_ACTION_PENALTY = -0.2
    # MAX_POSITION_VISITS = 3

    PICKUP_GOOD_OBJECT = 50.0           # Good reward, but not too high
    PICKUP_CORRECT_OBJECT = 8.0      # Strong but not nuclear penalty
    PICKUP_INVALID = -15.0             # Meaningful penalty but not crippling
    
    # === DROP REWARDS ===
    DROP_CORRECT_ZONE = 25.0           # Main reward - higher than pickup
    DROP_WRONG_ZONE = -3.0             # Small penalty - encourage trying
    DROP_INVALID = -8.0                # Moderate penalty
    
    # === MOVEMENT REWARDS ===
    MOVEMENT_INVALID = -10            # Small wall bump penalty
    DISTANCE_SHAPING_SCALE = 3       # Moderate movement guidance
    OBJECT_SEEKING_SCALE = 1.0 
    
    # === EPISODE REWARDS ===
    STEP_PENALTY = -0.02               # Small efficiency pressure
    COMPLETION_BONUS = 150.0           # Strong completion incentive
    
    # Anti-exploitation
    ENABLE_ANTI_LOOP = False
    LOOP_PENALTY = -0.5
    REPEAT_ACTION_PENALTY = -0.2
    MAX_POSITION_VISITS = 3
    