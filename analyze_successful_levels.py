#!/usr/bin/env python3
"""
Analyze successful levels to understand what works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level, check_win, advance_game_state, Direction
import json

def main():
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Analyze successful levels
    successful_level_ids = ['12', '30', '38', '39', '44']
    
    print("ANALYSIS OF SUCCESSFUL LEVELS:")
    
    for level in levels:
        if level['id'] in successful_level_ids:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            print(f"\n=== LEVEL {level_id} (SUCCESSFUL) ===")
            print(f"Expected solution: {expected_solution} ({len(expected_solution)} moves)")
            
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            print(f"Rules: {initial_state.rules}")
            print(f"Players: {len(initial_state.players) if initial_state.players else 0}")
            print(f"Winnables: {len(initial_state.winnables) if initial_state.winnables else 0}")
            
            # Test A* heuristic
            agent = ASTAR_CHATAgent()
            initial_heuristic = agent._calculate_heuristic(initial_state)
            rule_cost = agent._heuristic_rules(initial_state)
            distance_cost = agent._heuristic_distance(initial_state)
            
            print(f"Initial heuristic: {initial_heuristic}")
            print(f"Rule cost: {rule_cost}")
            print(f"Distance cost: {distance_cost}")

if __name__ == "__main__":
    main()
