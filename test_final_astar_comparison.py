#!/usr/bin/env python3
"""
Final comparison test - Original vs Improved A* agent
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.minimal_astar_AGENT import MinimalASTARAgent
from baba import parse_map, GameState, make_level
import json
import time

def final_comparison():
    """Final comparison between original and improved agent"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Test on first 20 levels
    test_levels = levels[:20]
    
    print("FINAL A* AGENT COMPARISON")
    print("=" * 60)
    
    agents = [
        ("Original", MinimalASTARAgent()),  # The original working version
        ("Improved", ASTAR_CHATAgent())     # The current improved version
    ]
    
    results = {}
    
    for agent_name, agent in agents:
        print(f"\nTesting {agent_name} Agent:")
        results[agent_name] = {'successes': 0, 'solutions': {}, 'times': {}}
        
        for level in test_levels:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            try:
                # Parse the level
                map_data = parse_map(ascii_map)
                initial_state = make_level(map_data)
                
                # Time the search - use appropriate iterations for each agent
                iterations = 250 if agent_name == "Original" else 500
                start_time = time.time()
                solution = agent.search(initial_state, iterations)
                search_time = time.time() - start_time
                
                if solution:
                    results[agent_name]['successes'] += 1
                    results[agent_name]['solutions'][level_id] = len(solution)
                    results[agent_name]['times'][level_id] = search_time
                    print(f"  Level {level_id}: ✓ {len(solution)} moves ({search_time:.2f}s)")
                else:
                    print(f"  Level {level_id}: ✗ No solution ({search_time:.2f}s)")
                    
            except Exception as e:
                print(f"  Level {level_id}: ✗ Error: {e}")
    
    # Summary comparison
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS:")
    
    for agent_name in results:
        successes = results[agent_name]['successes']
        total_levels = len(test_levels)
        print(f"\n{agent_name} Agent:")
        print(f"  Success rate: {successes}/{total_levels} ({successes/total_levels*100:.1f}%)")
        
        if results[agent_name]['solutions']:
            avg_solution_length = sum(results[agent_name]['solutions'].values()) / len(results[agent_name]['solutions'])
            avg_time = sum(results[agent_name]['times'].values()) / len(results[agent_name]['times'])
            print(f"  Average solution length: {avg_solution_length:.1f} moves")
            print(f"  Average search time: {avg_time:.2f} seconds")
    
    # Show improvements
    original_solved = set(results.get('Original', {}).get('solutions', {}).keys())
    improved_solved = set(results.get('Improved', {}).get('solutions', {}).keys())
    
    improvements = improved_solved - original_solved
    regressions = original_solved - improved_solved
    
    print(f"\nCOMPARISON ANALYSIS:")
    if improvements:
        print(f"✓ Levels newly solved by Improved agent: {sorted(improvements)}")
    if regressions:
        print(f"✗ Levels lost by Improved agent: {sorted(regressions)}")
    
    net_improvement = len(improvements) - len(regressions)
    print(f"Net improvement: {'+' if net_improvement >= 0 else ''}{net_improvement} levels")
    
    # Show efficiency improvements on common levels
    common_solved = original_solved & improved_solved
    better_solutions = []
    worse_solutions = []
    
    for level_id in common_solved:
        orig_len = results['Original']['solutions'][level_id]
        improved_len = results['Improved']['solutions'][level_id]
        
        if improved_len < orig_len:
            better_solutions.append((level_id, orig_len, improved_len))
        elif improved_len > orig_len:
            worse_solutions.append((level_id, orig_len, improved_len))
    
    if better_solutions:
        print(f"\n✓ Better solutions found by Improved agent:")
        for level_id, orig_len, improved_len in better_solutions:
            print(f"  Level {level_id}: {orig_len} → {improved_len} moves")
    
    if worse_solutions:
        print(f"\n✗ Worse solutions by Improved agent:")
        for level_id, orig_len, improved_len in worse_solutions:
            print(f"  Level {level_id}: {orig_len} → {improved_len} moves")

if __name__ == "__main__":
    final_comparison()
