# Zoning Environment - Reinforcement Learning
A custom reinforcement learning environment where an agent learns to place objects in their correct zones using PPO (Proximal Policy Optimization).

## 🎯 Task Description
The agent operates on different sized grids:
- Places red objects in red zone (left half of grid)
- Places blue objects in blue zone (right half of grid)

**Actions Available:**
- 0-3: Movement (Up, Down, Left, Right)
- 4: Pickup object at current position
- 5: Drop carried object at current position

## 🚀 Quick Start
(uv will automatically download the dependencies needed to run a file)
1. uv run transfer_learning.py, choose 1 - Trains and evaluates each stage of the transfer learning curriculum and saves the models in /models with logs in /logs
4. video_recorder.py - Creates gifs of the best runs using all the models in /models and saves them in /recordings 



## Notes for this version
Models have trained the best so far with just completion as reward
🏆 4x4_1obj       : 100.0% success |  102.4 reward |   6.2 steps
✅ 4x4_2obj       : 75.0% success |   47.3 reward |  51.6 steps
🏆 5x5_1obj       : 85.0% success |   63.5 reward |  29.8 steps
🏆 5x5_2obj       : 95.0% success |   85.7 reward |  25.4 steps