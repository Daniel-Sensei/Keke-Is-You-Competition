#!/usr/bin/env python3
"""
Test advanced A* agent on problematic levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.advanced_astar_AGENT import AdvancedASTARAgent
from baba import parse_map, GameState, make_level
import json
import time

def test_advanced_vs_original():
    """Test advanced agent vs original on problematic levels"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Focus on problematic levels first, then test broader
    test_level_ids = ['1', '6', '15', '40', '67', '70', '72', '73']
    
    print("TESTING ADVANCED A* vs ORIGINAL A*")
    print("=" * 60)
    
    original_agent = ASTAR_CHATAgent()
    advanced_agent = AdvancedASTARAgent()
    
    results = {'original': {}, 'advanced': {}}
    
    for level_data in levels:
        if level_data['id'] in test_level_ids:
            level_id = level_data['id']
            ascii_map = level_data['ascii']
            expected_solution = level_data['solution']
            
            print(f"\n=== LEVEL {level_id} ===")
            print(f"Expected: {len(expected_solution)} moves")
            
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            # Test both agents
            for agent_name, agent in [("Original", original_agent), ("Advanced", advanced_agent)]:
                try:
                    start_time = time.time()
                    solution = agent.search(initial_state.copy(), 800)
                    search_time = time.time() - start_time
                    
                    if solution:
                        efficiency = len(solution) / len(expected_solution)
                        print(f"{agent_name}: ✓ {len(solution)} moves ({search_time:.2f}s, eff: {efficiency:.2f})")
                        results[agent_name.lower()][level_id] = {
                            'solved': True,
                            'moves': len(solution),
                            'time': search_time,
                            'efficiency': efficiency
                        }
                    else:
                        print(f"{agent_name}: ✗ No solution ({search_time:.2f}s)")
                        results[agent_name.lower()][level_id] = {
                            'solved': False,
                            'time': search_time
                        }
                except Exception as e:
                    print(f"{agent_name}: ✗ Error: {e}")
                    results[agent_name.lower()][level_id] = {
                        'solved': False,
                        'error': str(e)
                    }
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    
    for agent_name in ['original', 'advanced']:
        solved = sum(1 for result in results[agent_name].values() if result.get('solved', False))
        total = len(results[agent_name])
        print(f"{agent_name.title()}: {solved}/{total} levels solved")
    
    # Show improvements
    improvements = []
    for level_id in test_level_ids:
        orig_solved = results['original'].get(level_id, {}).get('solved', False)
        adv_solved = results['advanced'].get(level_id, {}).get('solved', False)
        
        if adv_solved and not orig_solved:
            improvements.append(level_id)
    
    if improvements:
        print(f"\nLevels newly solved by Advanced agent: {improvements}")
    else:
        print(f"\nNo new levels solved by Advanced agent")

def test_full_comparison():
    """Test on all training levels for comprehensive comparison"""
    print(f"\n" + "="*60)
    print("FULL TRAINING SET COMPARISON")
    
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels'][:25]  # Test first 25 levels
    
    original_agent = ASTAR_CHATAgent()
    advanced_agent = AdvancedASTARAgent()
    
    original_successes = 0
    advanced_successes = 0
    
    for level_data in levels:
        level_id = level_data['id']
        ascii_map = level_data['ascii']
        
        # Parse level
        map_data = parse_map(ascii_map)
        initial_state = make_level(map_data)
        
        # Test both agents with generous iterations
        orig_solution = original_agent.search(initial_state.copy(), 500)
        adv_solution = advanced_agent.search(initial_state.copy(), 500)
        
        orig_solved = len(orig_solution) > 0
        adv_solved = len(adv_solution) > 0
        
        if orig_solved:
            original_successes += 1
        if adv_solved:
            advanced_successes += 1
        
        # Show status
        orig_status = "✓" if orig_solved else "✗"
        adv_status = "✓" if adv_solved else "✗"
        
        improvement = ""
        if adv_solved and not orig_solved:
            improvement = " 🎯"
        elif orig_solved and not adv_solved:
            improvement = " ⚠"
        
        print(f"Level {level_id:2}: Orig {orig_status} | Adv {adv_status}{improvement}")
    
    print(f"\nFINAL RESULTS:")
    print(f"Original agent: {original_successes}/{len(levels)} ({original_successes/len(levels)*100:.1f}%)")
    print(f"Advanced agent: {advanced_successes}/{len(levels)} ({advanced_successes/len(levels)*100:.1f}%)")
    
    net_improvement = advanced_successes - original_successes
    print(f"Net improvement: {'+' if net_improvement >= 0 else ''}{net_improvement} levels")

if __name__ == "__main__":
    test_advanced_vs_original()
    test_full_comparison()
