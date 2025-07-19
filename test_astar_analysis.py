#!/usr/bin/env python3
"""
Test script to analyze A* agent performance on training levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level
import json

def load_training_levels():
    """Load training levels from JSON file"""
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    return data['levels']

def test_astar_on_sample_levels():
    """Test A* agent on a sample of training levels to identify failure patterns"""
    agent = ASTAR_CHATAgent()
    levels = load_training_levels()
    
    # Test on first 10 levels for detailed analysis
    sample_levels = levels[:10]
    
    print("Testing A* agent on sample training levels:")
    print("=" * 50)
    
    successes = 0
    failures = []
    
    for i, level in enumerate(sample_levels):
        level_id = level['id']
        ascii_map = level['ascii']
        expected_solution = level['solution']
        
        print(f"\nLevel {level_id}:")
        print(f"Expected solution length: {len(expected_solution)}")
        
        try:
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            # Test with different iteration limits
            for iterations in [50, 100, 200]:
                solution = agent.search(initial_state, iterations)
                
                if solution:
                    print(f"  Found solution with {iterations} iterations: {len(solution)} moves")
                    # Convert solution to string format
                    solution_str = ''.join([d.value for d in solution])
                    
                    # Check if solution is correct (simple length comparison for now)
                    if len(solution_str) <= len(expected_solution) * 2:  # Allow some inefficiency
                        print(f"  ✓ Success! Solution: {solution_str}")
                        successes += 1
                        break
                    else:
                        print(f"  ⚠ Solution too long: {solution_str}")
                else:
                    print(f"  ✗ No solution found with {iterations} iterations")
            else:
                failures.append({
                    'level_id': level_id,
                    'expected_solution': expected_solution,
                    'ascii': ascii_map
                })
                print(f"  ✗ Failed to solve level {level_id}")
        
        except Exception as e:
            print(f"  ✗ Error processing level {level_id}: {e}")
            failures.append({
                'level_id': level_id,
                'expected_solution': expected_solution,
                'ascii': ascii_map,
                'error': str(e)
            })
    
    print(f"\n" + "=" * 50)
    print(f"Results: {successes}/{len(sample_levels)} levels solved")
    
    if failures:
        print(f"\nFailed levels:")
        for failure in failures:
            print(f"  Level {failure['level_id']}: {failure.get('error', 'No solution found')}")
    
    return successes, failures

if __name__ == "__main__":
    successes, failures = test_astar_on_sample_levels()
