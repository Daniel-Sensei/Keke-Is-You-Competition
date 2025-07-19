#!/usr/bin/env python3
"""
Detailed analysis of failed levels for A* agent
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from baba import parse_map, GameState, make_level, check_win, advance_game_state, Direction
import json

def analyze_level(level_data):
    """Analyze a specific level in detail"""
    level_id = level_data['id']
    ascii_map = level_data['ascii']
    expected_solution = level_data['solution']
    
    print(f"\n=== LEVEL {level_id} ANALYSIS ===")
    print(f"ASCII Map:")
    print(ascii_map)
    print(f"Expected solution: {expected_solution} ({len(expected_solution)} moves)")
    
    # Parse the level
    map_data = parse_map(ascii_map)
    initial_state = make_level(map_data)
    
    print(f"\nInitial state:")
    print(f"  Players: {len(initial_state.players) if initial_state.players else 0}")
    print(f"  Rules: {initial_state.rules}")
    print(f"  Winnables: {len(initial_state.winnables) if initial_state.winnables else 0}")
    print(f"  Words: {len(initial_state.words) if initial_state.words else 0}")
    print(f"  Keywords: {len(initial_state.keywords) if initial_state.keywords else 0}")
    
    # Test the expected solution
    test_state = initial_state.copy()
    moves = []
    for char in expected_solution:
        direction = Direction(char.lower())
        if direction != Direction.Undefined:
            moves.append(direction)
            test_state = advance_game_state(direction, test_state)
            if check_win(test_state):
                print(f"  Expected solution works! Wins after {len(moves)} moves")
                break
    else:
        print(f"  Expected solution verification failed or doesn't reach win state")
    
    # Test A* with detailed output
    agent = ASTAR_CHATAgent()
    print(f"\nA* Analysis:")
    
    # Test heuristic value
    initial_heuristic = agent._calculate_heuristic(initial_state)
    print(f"  Initial heuristic value: {initial_heuristic}")
    
    # Test rule formation
    rule_cost = agent._heuristic_rules(initial_state)
    distance_cost = agent._heuristic_distance(initial_state)
    print(f"  Rule formation cost: {rule_cost}")
    print(f"  Distance cost: {distance_cost}")
    
    # Check if there are viable "you" rules
    you_rules = agent._get_viable_you_rules(initial_state)
    print(f"  Viable 'you' rules: {you_rules}")
    
    return {
        'level_id': level_id,
        'initial_heuristic': initial_heuristic,
        'rule_cost': rule_cost,
        'distance_cost': distance_cost,
        'you_rules': you_rules,
        'has_players': len(initial_state.players) > 0 if initial_state.players else False,
        'has_winnables': len(initial_state.winnables) > 0 if initial_state.winnables else False
    }

def main():
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Analyze failed levels from previous test
    failed_level_ids = ['1', '6', '15', '40']
    
    analyses = []
    for level in levels:
        if level['id'] in failed_level_ids:
            analysis = analyze_level(level)
            analyses.append(analysis)
    
    print(f"\n" + "="*60)
    print("SUMMARY OF FAILED LEVELS:")
    for analysis in analyses:
        print(f"Level {analysis['level_id']}:")
        print(f"  Initial heuristic: {analysis['initial_heuristic']}")
        print(f"  Rule cost: {analysis['rule_cost']}")
        print(f"  Distance cost: {analysis['distance_cost']}")
        print(f"  Has players: {analysis['has_players']}")
        print(f"  Has winnables: {analysis['has_winnables']}")
        print(f"  Viable you rules: {analysis['you_rules']}")

if __name__ == "__main__":
    main()
