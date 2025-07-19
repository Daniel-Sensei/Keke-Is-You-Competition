"""
Test the robust hybrid agent on the first few levels.
"""

import json
import time
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from baba import GameState, Direction, make_level, advance_game_state, check_win, parse_map
from agents.robust_hybrid_AGENT import ROBUSTHYBRIDAgent

def test_level(agent, level_data, level_num):
    """Test the agent on a single level."""
    print(f"\n{'='*60}")
    print(f"Testing Level {level_num}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Convert level data to GameState
        ascii_map = level_data['ascii']
        parsed_map = parse_map(ascii_map)
        level = make_level(parsed_map)
        
        # Test the agent
        solution = agent.search(level, iterations=100)
        elapsed_time = time.time() - start_time
        
        # Verify solution
        solved = False
        if solution:
            try:
                current_state = level
                for action in solution:
                    current_state = advance_game_state(action, current_state.copy())
                    if check_win(current_state):
                        solved = True
                        break
            except Exception as e:
                print(f"Error validating solution: {e}")
                solved = False
        
        print(f"Result: {'SOLVED' if solved else 'FAILED'}")
        print(f"Solution length: {len(solution) if solution else 0}")
        print(f"Time: {elapsed_time:.2f}s")
        
        if solution:
            print(f"Solution: {[action.value for action in solution]}")
            
        # Compare with expected solution if available
        if 'solution' in level_data and level_data['solution']:
            expected = level_data['solution']
            print(f"Expected: {expected} (length: {len(expected)})")
            
        return solved, len(solution) if solution else 0, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR: {e}")
        return False, 0, elapsed_time

def main():
    # Load levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
        train_levels = data.get('levels', [])
    
    print(f"Loaded {len(train_levels)} train levels")
    
    # Create robust hybrid agent
    agent = ROBUSTHYBRIDAgent(max_time_per_phase=30.0)
    
    # Test first 5 levels
    results = []
    for i in range(min(5, len(train_levels))):
        level_data = train_levels[i]
        solved, length, time_taken = test_level(agent, level_data, i+1)
        results.append((i+1, solved, length, time_taken))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    solved_count = sum(1 for _, solved, _, _ in results if solved)
    total_time = sum(time_taken for _, _, _, time_taken in results)
    
    print(f"Levels solved: {solved_count}/{len(results)}")
    print(f"Total time: {total_time:.2f}s")
    
    for level_num, solved, length, time_taken in results:
        status = "SOLVED" if solved else "FAILED"
        print(f"Level {level_num}: {status} ({length} moves, {time_taken:.2f}s)")

if __name__ == "__main__":
    main()
