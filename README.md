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
I ran grid_size_curriculum.py to train the model. Went from 4x4 -> 5x5 -> 6x6 -> 7x7 -> 8x8 -> 9x9 -> 10x10, 2 zones, 1 object in an incorrect zone. Reward for task completion and potential shaping to move towards object/zone.

🏆 CURRICULUM SUMMARY
==================================================
🏆 4x4: 100.0% success rate
🏆 5x5: 100.0% success rate
🏆 6x6: 93.3% success rate
🏆 7x7: 100.0% success rate
❌ 8x8: 33.3% success rate
🏆 9x9: 93.3% success rate
⚠️ 10x10: 46.7% success rate

The agent is able to optimize for larger grids than it was without potential shaping but still starts to fail after a bit. 

