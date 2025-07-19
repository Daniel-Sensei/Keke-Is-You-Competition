#!/usr/bin/env python3
"""
Test the improved A* agent on the problematic levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level
import json
import time

def test_improved_agent():
    """Test the improved A* agent on all training levels"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Test on first 20 levels for comprehensive evaluation
    test_levels = levels[:20]
    
    print("TESTING IMPROVED A* AGENT")
    print("=" * 50)
    
    agent = ASTAR_CHATAgent()
    successes = 0
    total_time = 0
    
    results = {}
    
    for level in test_levels:
        level_id = level['id']
        ascii_map = level['ascii']
        expected_solution = level['solution']
        
        print(f"\nLevel {level_id} (expected: {len(expected_solution)} moves):", end=" ")
        
        try:
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            # Time the search
            start_time = time.time()
            solution = agent.search(initial_state, 400)  # Higher iteration limit
            search_time = time.time() - start_time
            total_time += search_time
            
            if solution:
                successes += 1
                solution_str = ''.join([d.value for d in solution])
                efficiency = len(solution) / len(expected_solution) if len(expected_solution) > 0 else 1.0
                
                print(f"✓ {len(solution)} moves ({search_time:.2f}s) - {solution_str[:20]}{'...' if len(solution_str) > 20 else ''}")
                if efficiency <= 1.2:  # Good efficiency
                    print(f"    Efficiency: {efficiency:.2f} (Good!)")
                elif efficiency <= 2.0:  # Acceptable
                    print(f"    Efficiency: {efficiency:.2f} (OK)")
                else:  # Poor efficiency
                    print(f"    Efficiency: {efficiency:.2f} (Poor)")
                    
                results[level_id] = {
                    'solved': True,
                    'moves': len(solution),
                    'time': search_time,
                    'efficiency': efficiency,
                    'solution': solution_str
                }
            else:
                print(f"✗ No solution ({search_time:.2f}s)")
                results[level_id] = {
                    'solved': False,
                    'time': search_time
                }
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results[level_id] = {
                'solved': False,
                'error': str(e)
            }
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS SUMMARY:")
    print(f"Success rate: {successes}/{len(test_levels)} ({successes/len(test_levels)*100:.1f}%)")
    print(f"Total search time: {total_time:.2f} seconds")
    print(f"Average time per level: {total_time/len(test_levels):.2f} seconds")
    
    # Show solved levels
    solved_levels = [lid for lid, result in results.items() if result.get('solved', False)]
    failed_levels = [lid for lid, result in results.items() if not result.get('solved', False)]
    
    print(f"\nSolved levels: {sorted(solved_levels)}")
    if failed_levels:
        print(f"Failed levels: {sorted(failed_levels)}")
    
    # Show efficiency statistics
    efficiencies = [result['efficiency'] for result in results.values() if result.get('solved') and 'efficiency' in result]
    if efficiencies:
        avg_efficiency = sum(efficiencies) / len(efficiencies)
        print(f"\nAverage solution efficiency: {avg_efficiency:.2f}")
        good_solutions = len([e for e in efficiencies if e <= 1.2])
        print(f"Solutions with good efficiency (≤1.2): {good_solutions}/{len(efficiencies)}")

if __name__ == "__main__":
    test_improved_agent()
