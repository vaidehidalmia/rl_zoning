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


## Specific to this version notes
Improved code for environment, train, evaluate, video_recorder. Now uses config and utils.


