#!/usr/bin/env python3
"""Test current A* agent performance on problematic levels"""

import json
from baba import parse_map, make_level
from agents.astar_chat_AGENT import ASTAR_CHATAgent
import time

def test_specific_levels():
    """Test specific problematic levels"""
    problematic_levels = [1, 6, 15, 40]
    
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
        all_levels = data['levels']
    
    agent = ASTAR_CHATAgent()
    results = []
    
    print("Testing current A* agent on problematic levels:")
    print("=" * 60)
    
    for target_id in [str(i) for i in problematic_levels]:
        level = next((l for l in all_levels if l['id'] == target_id), None)
        if level:
            print(f"\nTesting Level {target_id}:")
            
            # Parse level
            map_data = parse_map(level['ascii'])
            initial_state = make_level(map_data)
            
            start_time = time.time()
            solution = agent.search(initial_state, 500)
            end_time = time.time()
            
            if solution:
                moves = len(solution)
                print(f"  ✓ SOLVED in {moves} moves ({end_time-start_time:.2f}s)")
                results.append((target_id, True, moves, end_time-start_time))
            else:
                print(f"  ✗ FAILED ({end_time-start_time:.2f}s)")
                results.append((target_id, False, 0, end_time-start_time))
        else:
            print(f"Level {target_id} not found")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    solved = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    print(f"Solved: {solved}/{total} levels")
    
    for level_id, success, moves, time_taken in results:
        status = "SOLVED" if success else "FAILED"
        if success:
            print(f"  Level {level_id}: {status} ({moves} moves, {time_taken:.2f}s)")
        else:
            print(f"  Level {level_id}: {status} ({time_taken:.2f}s)")
    
    return results

def test_batch_performance():
    """Test performance on first 30 levels"""
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
        all_levels = data['levels']
    
    test_levels = all_levels[:30]
    agent = ASTAR_CHATAgent()
    
    print("\nTesting batch performance (first 30 levels):")
    print("=" * 60)
    
    solved = 0
    failed_levels = []
    
    start_time = time.time()
    
    for i, level in enumerate(test_levels):
        try:
            map_data = parse_map(level['ascii'])
            initial_state = make_level(map_data)
            
            solution = agent.search(initial_state, 300)  # Shorter iterations for batch
            
            if solution:
                solved += 1
                print(f"Level {level['id']}: ✓")
            else:
                failed_levels.append(level['id'])
                print(f"Level {level['id']}: ✗")
                
        except Exception as e:
            failed_levels.append(level['id'])
            print(f"Level {level['id']}: ✗ (Error: {e})")
    
    end_time = time.time()
    total = len(test_levels)
    
    print(f"\nResults: {solved}/{total} levels solved ({end_time-start_time:.2f}s)")
    print(f"Failed levels: {failed_levels}")
    
    return solved, total

if __name__ == "__main__":
    # Test specific problematic levels
    specific_results = test_specific_levels()
    
    # Test batch performance
    batch_solved, batch_total = test_batch_performance()
    
    print(f"\nOVERALL RESULTS:")
    print(f"Problematic levels: {sum(1 for _, success, _, _ in specific_results if success)}/{len(specific_results)}")
    print(f"Batch performance: {batch_solved}/{batch_total}")
