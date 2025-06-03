# Zoning Environment - Reinforcement Learning
A custom reinforcement learning environment where an agent learns to place objects in their correct zones using PPO (Proximal Policy Optimization).

## 🎯 Task Description
The agent operates on a 6x6 grid and must:
- Place red objects in red zone (left half of grid)
- Place blue objects in blue zone (right half of grid)

**Actions Available:**
- 0-3: Movement (Up, Down, Left, Right)
- 4: Pickup object at current position
- 5: Drop carried object at current position

## 🚀 Quick Start
1. Install Dependencies - pip install gymnasium stable-baselines3 matplotlib numpy
2. Train the Agent - python train.py
3. Evaluate Performance - python evaluate.py
4. Create gifs of some runs/best run - video_recorder.py

## 🎨 Visualization
After running video_recorder.py, the gifs are saved in /recordings



# Specific to this version notes
I ran grid_size_curriculum.py to train the model. Went from 4x4 -> 5x5 -> 6x6 -> 7x7 -> 8x8, 2 zones, 1 object in an incorrect zone. Only reward is for task completion.

🏆 CURRICULUM SUMMARY
==================================================
🏆 4x4: 86.7% success rate
🏆 5x5: 93.3% success rate
🏆 6x6: 80.0% success rate
⚠️ 7x7: 46.7% success rate
❌ 8x8: 0.0% success rate

The agent isn't able to optimize its search with larger grid sizes. Using debug_test, I learnt that the agent learns that pickup/drop are useless actions to take.

## Options for improvement:
Option 1: Add relative position features
relative_x = object_pos[0] - agent_pos[0]  
relative_y = object_pos[1] - agent_pos[1]
obs_enhanced = np.concatenate([obs.flatten(), [relative_x, relative_y]])

Option 2: Curriculum with position rewards
if agent_gets_closer_to_object:
    reward += small_bonus
(need to be careful as it is easily exploitable)

