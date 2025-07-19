#!/usr/bin/env python3
"""
Test current state and analyze specific problematic levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level, advance_game_state, Direction, check_win
import json

def test_problematic_levels():
    """Test the specific levels that are failing"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Focus on problematic levels
    problematic_ids = ['1', '6', '15', '40']
    
    print("ANALYZING PROBLEMATIC LEVELS")
    print("=" * 50)
    
    agent = ASTAR_CHATAgent()
    
    for level_data in levels:
        if level_data['id'] in problematic_ids:
            level_id = level_data['id']
            ascii_map = level_data['ascii']
            expected_solution = level_data['solution']
            
            print(f"\n=== LEVEL {level_id} ===")
            print(f"Expected: {expected_solution} ({len(expected_solution)} moves)")
            
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            print(f"Initial rules: {initial_state.rules}")
            print(f"Players: {len(initial_state.players)}")
            print(f"Winnables: {len(initial_state.winnables)}")
            
            # Test current agent
            for iterations in [200, 400, 600]:
                solution = agent.search(initial_state.copy(), iterations)
                if solution:
                    print(f"✓ Solved with {iterations} iterations: {len(solution)} moves")
                    break
                else:
                    print(f"✗ Failed with {iterations} iterations")
            
            # Analyze heuristic
            h_value = agent._calculate_heuristic(initial_state)
            rule_cost = agent._heuristic_rules(initial_state)
            dist_cost = agent._heuristic_distance(initial_state)
            print(f"Heuristic: {h_value:.2f} (rules: {rule_cost:.2f}, dist: {dist_cost:.2f})")

if __name__ == "__main__":
    test_problematic_levels()
