"""
Debug why 8x8 fails when 6x6 works

Let's see exactly what the agent is doing wrong on large grids
"""

from stable_baselines3 import PPO
from environment import ZoningEnv
import numpy as np

def debug_agent_behavior(grid_size, model_path, episodes=3):
    """Detailed analysis of agent behavior"""
    print(f"\n🔍 DEBUGGING {grid_size}x{grid_size} AGENT BEHAVIOR")
    print("=" * 60)
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except:
        print(f"❌ Could not load {model_path}")
        return
    
    action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Pickup", 5: "Drop"}
    
    for episode in range(episodes):
        print(f"\n{'='*40}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*40}")
        
        env = ZoningEnv(grid_size=grid_size, num_objects=1)
        env.max_steps = 400  # Generous time limit
        
        obs, _ = env.reset()
        print(f"🎮 Initial state:")
        print(f"   Agent at: {env.agent_pos}")
        print(f"   Object at: {list(env.objects.keys())[0]}")
        print(f"   Distance: {calculate_distance(env.agent_pos, list(env.objects.keys())[0])}")
        
        # Track behavior patterns
        visited_positions = set()
        pickup_attempts = 0
        action_sequence = []
        position_history = []
        
        for step in range(50):  # First 50 steps analysis
            action, _ = model.predict(obs, deterministic=False)
            action_name = action_names.get(int(action), "Unknown")
            
            # Track patterns
            visited_positions.add(env.agent_pos)
            position_history.append(env.agent_pos)
            action_sequence.append(int(action))
            
            if action == 4:  # Pickup attempt
                pickup_attempts += 1
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if step < 10 or step % 10 == 0:  # Show first 10 steps, then every 10th
                print(f"Step {step+1:2d}: {action_name:6s} → {env.agent_pos} → Carrying: {env.carried_object}")
            
            if terminated:
                print(f"\n🎯 SUCCESS! Completed in {step+1} steps")
                break
            elif truncated:
                print(f"\n⏰ TIMEOUT after {step+1} steps")
                break
        
        # Analyze behavior patterns
        print(f"\n📊 BEHAVIOR ANALYSIS:")
        print(f"   Positions visited: {len(visited_positions)}/{grid_size**2} ({len(visited_positions)/(grid_size**2)*100:.1f}%)")
        print(f"   Pickup attempts: {pickup_attempts}")
        print(f"   Final distance to object: {calculate_distance(env.agent_pos, list(env.objects.keys())[0])}")
        
        # Movement pattern analysis
        if len(position_history) >= 10:
            print(f"   Movement pattern analysis:")
            movements = [(position_history[i+1][0] - position_history[i][0], 
                         position_history[i+1][1] - position_history[i][1]) 
                        for i in range(min(10, len(position_history)-1))]
            unique_moves = len(set(movements))
            print(f"     Unique movement directions in first 10: {unique_moves}/10")
            print(f"     {'Systematic' if unique_moves >= 6 else 'Repetitive'} exploration detected")
        
        # Action distribution
        action_counts = {name: action_sequence.count(i) for i, name in action_names.items()}
        print(f"   Action distribution: {action_counts}")

def calculate_distance(pos1, pos2):
    """Manhattan distance between positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def compare_grid_performance():
    """Compare successful vs failing grid sizes"""
    print("\n🎯 COMPARING SUCCESSFUL vs FAILING PERFORMANCE")
    print("=" * 60)
    
    # Test configurations: (grid_size, model_path, expected_success)
    tests = [
        (6, "models/grid_6x6", "Should work well"),
        (7, "models/grid_7x7", "Moderate performance"), 
        (8, "models/grid_8x8", "Fails completely")
    ]
    
    for grid_size, model_path, expectation in tests:
        print(f"\n{expectation}: {grid_size}x{grid_size}")
        debug_agent_behavior(grid_size, model_path, episodes=1)

def analyze_search_efficiency():
    """Analyze how efficiently agents search different grid sizes"""
    print("\n🔍 SEARCH EFFICIENCY ANALYSIS")
    print("=" * 40)
    
    for grid_size in [4, 6, 8]:
        print(f"\n{grid_size}x{grid_size} Search Requirements:")
        total_cells = grid_size ** 2
        avg_distance = grid_size  # Rough average
        
        print(f"   Total cells: {total_cells}")
        print(f"   Average distance: ~{avg_distance} steps")
        print(f"   Worst case search: {total_cells} cells to visit")
        print(f"   Time pressure: {400} step limit")
        print(f"   Search efficiency needed: {total_cells/400*100:.1f}% of limit")
        
        if total_cells/400 > 0.3:
            print(f"   ⚠️  High search pressure - needs systematic exploration")
        else:
            print(f"   ✅ Manageable search space")

if __name__ == "__main__":
    # Run comprehensive debugging
    compare_grid_performance()
    analyze_search_efficiency()
    
    print(f"\n💡 INSIGHTS:")
    print("1. Check if agent is exploring systematically or randomly")
    print("2. See if pickup attempts are concentrated or spread out") 
    print("3. Analyze if movement patterns are efficient or repetitive")
    print("4. Compare successful (6x6) vs failed (8x8) strategies")