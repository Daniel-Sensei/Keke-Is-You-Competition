#!/usr/bin/env python3
"""
Ultimate test of optimal A* agent focusing on difficult levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.optimal_astar_AGENT import OptimalASTARAgent
from baba import parse_map, GameState, make_level
import json
import time

def test_difficult_levels():
    """Focus specifically on the levels that consistently fail"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Focus on the most challenging levels based on previous analysis
    difficult_level_ids = ['1', '6', '15', '40', '67', '70', '72', '73', '82', '88']
    
    print("TESTING OPTIMAL A* ON DIFFICULT LEVELS")
    print("=" * 60)
    
    original_agent = ASTAR_CHATAgent()
    optimal_agent = OptimalASTARAgent()
    
    original_results = {}
    optimal_results = {}
    
    for level in levels:
        if level['id'] in difficult_level_ids:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            print(f"\n=== LEVEL {level_id} ===")
            print(f"Expected: {len(expected_solution)} moves")
            
            try:
                # Parse the level
                map_data = parse_map(ascii_map)
                initial_state = make_level(map_data)
                
                # Test original agent
                start_time = time.time()
                original_solution = original_agent.search(initial_state.copy(), 400)
                original_time = time.time() - start_time
                
                # Test optimal agent
                start_time = time.time()
                optimal_solution = optimal_agent.search(initial_state.copy(), 400)
                optimal_time = time.time() - start_time
                
                original_solved = len(original_solution) > 0
                optimal_solved = len(optimal_solution) > 0
                
                print(f"Original: {'✓' if original_solved else '✗'} " + 
                      f"({len(original_solution)} moves, {original_time:.2f}s)")
                print(f"Optimal:  {'✓' if optimal_solved else '✗'} " + 
                      f"({len(optimal_solution)} moves, {optimal_time:.2f}s)")
                
                if optimal_solved and not original_solved:
                    print(f"🎯 NEW SOLVE! Optimal agent solved this level!")
                elif optimal_solved and original_solved:
                    if len(optimal_solution) < len(original_solution):
                        print(f"⚡ IMPROVEMENT: {len(original_solution)} → {len(optimal_solution)} moves")
                    elif len(optimal_solution) == len(original_solution):
                        print(f"✓ SAME QUALITY: Both found {len(optimal_solution)} move solution")
                elif original_solved and not optimal_solved:
                    print(f"⚠ REGRESSION: Lost solution from original agent")
                
                original_results[level_id] = {
                    'solved': original_solved,
                    'moves': len(original_solution) if original_solved else None,
                    'time': original_time
                }
                
                optimal_results[level_id] = {
                    'solved': optimal_solved,
                    'moves': len(optimal_solution) if optimal_solved else None,
                    'time': optimal_time
                }
                
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY OF DIFFICULT LEVELS:")
    
    original_successes = sum(1 for r in original_results.values() if r['solved'])
    optimal_successes = sum(1 for r in optimal_results.values() if r['solved'])
    
    print(f"Original agent: {original_successes}/{len(difficult_level_ids)} levels solved")
    print(f"Optimal agent:  {optimal_successes}/{len(difficult_level_ids)} levels solved")
    print(f"Net improvement: {optimal_successes - original_successes} levels")
    
    # Show specific improvements
    improvements = []
    regressions = []
    
    for level_id in difficult_level_ids:
        if level_id in original_results and level_id in optimal_results:
            orig = original_results[level_id]['solved']
            opt = optimal_results[level_id]['solved']
            
            if opt and not orig:
                improvements.append(level_id)
            elif orig and not opt:
                regressions.append(level_id)
    
    if improvements:
        print(f"\nNew levels solved: {improvements}")
    if regressions:
        print(f"Levels lost: {regressions}")

def test_all_levels_quick():
    """Quick test on all levels to see overall performance"""
    print(f"\n" + "=" * 60)
    print("QUICK TEST ON ALL TRAINING LEVELS")
    
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels'][:30]  # First 30 levels for speed
    
    original_agent = ASTAR_CHATAgent()
    optimal_agent = OptimalASTARAgent()
    
    original_successes = 0
    optimal_successes = 0
    
    for level in levels:
        level_id = level['id']
        ascii_map = level['ascii']
        
        try:
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            # Quick test with limited iterations
            original_solution = original_agent.search(initial_state.copy(), 200)
            optimal_solution = optimal_agent.search(initial_state.copy(), 200)
            
            if len(original_solution) > 0:
                original_successes += 1
            if len(optimal_solution) > 0:
                optimal_successes += 1
                
        except Exception:
            pass
    
    print(f"Quick results on first 30 levels:")
    print(f"Original: {original_successes}/30 ({original_successes/30*100:.1f}%)")
    print(f"Optimal:  {optimal_successes}/30 ({optimal_successes/30*100:.1f}%)")
    print(f"Improvement: {optimal_successes - original_successes} levels")

if __name__ == "__main__":
    test_difficult_levels()
    test_all_levels_quick()
