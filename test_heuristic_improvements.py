#!/usr/bin/env python3
"""
Test complete performance with improved heuristic
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.minimal_astar_AGENT import MinimalASTARAgent
from baba import parse_map, GameState, make_level
import json
import time

def test_heuristic_improvements():
    """Test complete performance with improved heuristic"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Test on first 25 levels for comprehensive evaluation
    test_levels = levels[:25]
    
    print("TESTING HEURISTIC IMPROVEMENTS")
    print("=" * 60)
    
    agents = [
        ("Original", MinimalASTARAgent()),
        ("Improved Heuristic", ASTAR_CHATAgent())
    ]
    
    results = {}
    
    for agent_name, agent in agents:
        print(f"\nTesting {agent_name}:")
        results[agent_name] = {'successes': 0, 'solutions': {}, 'times': {}}
        total_time = 0
        
        for level in test_levels:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            try:
                # Parse the level
                map_data = parse_map(ascii_map)
                initial_state = make_level(map_data)
                
                # Use same iterations for fair comparison
                start_time = time.time()
                solution = agent.search(initial_state, 300)
                search_time = time.time() - start_time
                total_time += search_time
                
                if solution:
                    results[agent_name]['successes'] += 1
                    results[agent_name]['solutions'][level_id] = len(solution)
                    results[agent_name]['times'][level_id] = search_time
                    
                    efficiency = len(solution) / len(expected_solution)
                    status = "✓"
                    if efficiency <= 1.2:
                        status += " (efficient)"
                    elif efficiency <= 2.0:
                        status += " (ok)"
                    else:
                        status += " (long)"
                    
                    print(f"  Level {level_id}: {status} {len(solution)} moves")
                else:
                    print(f"  Level {level_id}: ✗")
                    
            except Exception as e:
                print(f"  Level {level_id}: ✗ Error: {e}")
        
        print(f"Total time: {total_time:.2f}s, Avg: {total_time/len(test_levels):.2f}s")
    
    # Summary comparison
    print(f"\n" + "=" * 60)
    print("HEURISTIC IMPROVEMENT RESULTS:")
    
    for agent_name in results:
        successes = results[agent_name]['successes']
        total_levels = len(test_levels)
        print(f"\n{agent_name}:")
        print(f"  Success rate: {successes}/{total_levels} ({successes/total_levels*100:.1f}%)")
        
        if results[agent_name]['solutions']:
            avg_solution_length = sum(results[agent_name]['solutions'].values()) / len(results[agent_name]['solutions'])
            avg_time = sum(results[agent_name]['times'].values()) / len(results[agent_name]['times'])
            print(f"  Average solution length: {avg_solution_length:.1f} moves")
            print(f"  Average search time: {avg_time:.2f} seconds")
    
    # Show improvements
    original_solved = set(results.get('Original', {}).get('solutions', {}).keys())
    improved_solved = set(results.get('Improved Heuristic', {}).get('solutions', {}).keys())
    
    improvements = improved_solved - original_solved
    regressions = original_solved - improved_solved
    
    print(f"\nIMPROVEMENT ANALYSIS:")
    if improvements:
        print(f"✓ Newly solved levels: {sorted(improvements)}")
    if regressions:
        print(f"✗ Lost levels: {sorted(regressions)}")
    
    net_improvement = len(improvements) - len(regressions)
    print(f"Net improvement: {'+' if net_improvement >= 0 else ''}{net_improvement} levels")
    
    # Efficiency analysis on common levels
    common_solved = original_solved & improved_solved
    if common_solved:
        better_solutions = 0
        worse_solutions = 0
        
        for level_id in common_solved:
            orig_len = results['Original']['solutions'][level_id]
            improved_len = results['Improved Heuristic']['solutions'][level_id]
            
            if improved_len < orig_len:
                better_solutions += 1
            elif improved_len > orig_len:
                worse_solutions += 1
        
        print(f"Solution quality: {better_solutions} better, {worse_solutions} worse out of {len(common_solved)} common")

if __name__ == "__main__":
    test_heuristic_improvements()
