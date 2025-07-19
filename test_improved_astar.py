#!/usr/bin/env python3
"""
Test improved A* agent on training levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.improved_astar_chat_AGENT import ImprovedASTAR_CHATAgent
from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level
import json

def test_agent_comparison():
    """Compare original vs improved A* agent"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Test on first 15 levels for comprehensive analysis
    sample_levels = levels[:15]
    
    print("COMPARING ORIGINAL vs IMPROVED A* AGENT")
    print("=" * 60)
    
    original_agent = ASTAR_CHATAgent()
    improved_agent = ImprovedASTAR_CHATAgent()
    
    original_successes = 0
    improved_successes = 0
    
    results = []
    
    for i, level in enumerate(sample_levels):
        level_id = level['id']
        ascii_map = level['ascii']
        expected_solution = level['solution']
        
        print(f"\nLevel {level_id} (expected: {len(expected_solution)} moves):")
        
        try:
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            # Test original agent
            original_solution = original_agent.search(initial_state.copy(), 200)
            original_solved = len(original_solution) > 0
            if original_solved:
                original_successes += 1
            
            # Test improved agent  
            improved_solution = improved_agent.search(initial_state.copy(), 200)
            improved_solved = len(improved_solution) > 0
            if improved_solved:
                improved_successes += 1
            
            print(f"  Original: {'✓' if original_solved else '✗'} ({len(original_solution)} moves)")
            print(f"  Improved: {'✓' if improved_solved else '✗'} ({len(improved_solution)} moves)")
            
            # Check for improvements
            if improved_solved and not original_solved:
                print(f"  🎯 IMPROVEMENT: Solved by improved agent!")
            elif improved_solved and original_solved and len(improved_solution) < len(original_solution):
                print(f"  ⚡ BETTER: Shorter solution ({len(original_solution)} → {len(improved_solution)})")
            
            results.append({
                'level_id': level_id,
                'expected_length': len(expected_solution),
                'original_solved': original_solved,
                'original_length': len(original_solution) if original_solved else None,
                'improved_solved': improved_solved,
                'improved_length': len(improved_solution) if improved_solved else None
            })
            
        except Exception as e:
            print(f"  ✗ Error processing level {level_id}: {e}")
            results.append({
                'level_id': level_id,
                'expected_length': len(expected_solution),
                'original_solved': False,
                'improved_solved': False,
                'error': str(e)
            })
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS SUMMARY:")
    print(f"Original agent: {original_successes}/{len(sample_levels)} levels solved")
    print(f"Improved agent: {improved_successes}/{len(sample_levels)} levels solved")
    print(f"Improvement: +{improved_successes - original_successes} levels")
    
    # Show detailed improvements
    improvements = []
    for result in results:
        if result.get('improved_solved') and not result.get('original_solved'):
            improvements.append(result['level_id'])
    
    if improvements:
        print(f"\nLevels newly solved by improved agent: {improvements}")
    
    # Show efficiency improvements
    better_solutions = []
    for result in results:
        if (result.get('original_solved') and result.get('improved_solved') and 
            result.get('improved_length') and result.get('original_length') and
            result['improved_length'] < result['original_length']):
            better_solutions.append((result['level_id'], result['original_length'], result['improved_length']))
    
    if better_solutions:
        print(f"\nBetter solutions found:")
        for level_id, orig_len, imp_len in better_solutions:
            print(f"  Level {level_id}: {orig_len} → {imp_len} moves")

if __name__ == "__main__":
    test_agent_comparison()
