#!/usr/bin/env python3
"""
Final comprehensive test of all A* agent versions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.final_astar_AGENT import FinalASTARAgent
from baba import parse_map, GameState, make_level
import json
import time

def test_all_agents():
    """Test all A* versions on training levels"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Test on first 20 levels for comprehensive evaluation
    sample_levels = levels[:20]
    
    print("COMPREHENSIVE A* AGENT COMPARISON")
    print("=" * 70)
    
    agents = [
        ("Original", ASTAR_CHATAgent()),
        ("Final", FinalASTARAgent())
    ]
    
    results = {}
    
    for agent_name, agent in agents:
        print(f"\nTesting {agent_name} Agent:")
        results[agent_name] = {'successes': 0, 'solutions': {}, 'times': {}}
        
        for level in sample_levels:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            try:
                # Parse the level
                map_data = parse_map(ascii_map)
                initial_state = make_level(map_data)
                
                # Time the search
                start_time = time.time()
                solution = agent.search(initial_state, 500)  # Generous iterations
                search_time = time.time() - start_time
                
                if solution:
                    results[agent_name]['successes'] += 1
                    results[agent_name]['solutions'][level_id] = len(solution)
                    results[agent_name]['times'][level_id] = search_time
                    print(f"  Level {level_id}: ✓ {len(solution)} moves ({search_time:.2f}s)")
                else:
                    print(f"  Level {level_id}: ✗ No solution")
                    
            except Exception as e:
                print(f"  Level {level_id}: ✗ Error: {e}")
    
    # Summary comparison
    print(f"\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY:")
    
    for agent_name in results:
        successes = results[agent_name]['successes']
        total_levels = len(sample_levels)
        print(f"\n{agent_name} Agent:")
        print(f"  Success rate: {successes}/{total_levels} ({successes/total_levels*100:.1f}%)")
        
        if results[agent_name]['solutions']:
            avg_solution_length = sum(results[agent_name]['solutions'].values()) / len(results[agent_name]['solutions'])
            avg_time = sum(results[agent_name]['times'].values()) / len(results[agent_name]['times'])
            print(f"  Average solution length: {avg_solution_length:.1f} moves")
            print(f"  Average search time: {avg_time:.2f} seconds")
    
    # Show improvements
    original_solved = set(results.get('Original', {}).get('solutions', {}).keys())
    final_solved = set(results.get('Final', {}).get('solutions', {}).keys())
    
    improvements = final_solved - original_solved
    regressions = original_solved - final_solved
    
    if improvements:
        print(f"\nLevels newly solved by Final agent: {sorted(improvements)}")
    if regressions:
        print(f"Levels lost by Final agent: {sorted(regressions)}")
    
    # Show efficiency improvements
    common_solved = original_solved & final_solved
    better_solutions = []
    worse_solutions = []
    
    for level_id in common_solved:
        orig_len = results['Original']['solutions'][level_id]
        final_len = results['Final']['solutions'][level_id]
        
        if final_len < orig_len:
            better_solutions.append((level_id, orig_len, final_len))
        elif final_len > orig_len:
            worse_solutions.append((level_id, orig_len, final_len))
    
    if better_solutions:
        print(f"\nBetter solutions found by Final agent:")
        for level_id, orig_len, final_len in better_solutions:
            print(f"  Level {level_id}: {orig_len} → {final_len} moves")
    
    if worse_solutions:
        print(f"\nWorse solutions by Final agent:")
        for level_id, orig_len, final_len in worse_solutions:
            print(f"  Level {level_id}: {orig_len} → {final_len} moves")

if __name__ == "__main__":
    test_all_agents()
