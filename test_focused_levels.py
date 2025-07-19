#!/usr/bin/env python3
"""
Focused test on difficult levels with rule formation analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.astar_chat_AGENT import ASTAR_CHATAgent
from agents.enhanced_astar_AGENT import EnhancedASTARAgent
from baba import parse_map, GameState, make_level, advance_game_state, Direction, check_win
import json

def test_win_rule_formation():
    """Test specifically on levels that need win rule formation"""
    # Load training levels
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    levels = data['levels']
    
    # Focus on problematic levels
    target_levels = ['1', '6', '15', '40']
    
    print("TESTING WIN RULE FORMATION ON DIFFICULT LEVELS")
    print("=" * 60)
    
    original_agent = ASTAR_CHATAgent()
    enhanced_agent = EnhancedASTARAgent()
    
    for level in levels:
        if level['id'] in target_levels:
            level_id = level['id']
            ascii_map = level['ascii']
            expected_solution = level['solution']
            
            print(f"\n=== LEVEL {level_id} ===")
            print(f"Expected solution: {expected_solution} ({len(expected_solution)} moves)")
            
            # Parse the level
            map_data = parse_map(ascii_map)
            initial_state = make_level(map_data)
            
            print(f"Initial rules: {initial_state.rules}")
            print(f"Players: {len(initial_state.players) if initial_state.players else 0}")
            print(f"Winnables: {len(initial_state.winnables) if initial_state.winnables else 0}")
            
            # Test both agents with increased iterations
            for agent_name, agent in [("Original", original_agent), ("Enhanced", enhanced_agent)]:
                for iterations in [300, 500]:
                    solution = agent.search(initial_state.copy(), iterations)
                    if solution:
                        print(f"{agent_name} ({iterations} iter): ✓ {len(solution)} moves - {''.join([d.value for d in solution])}")
                        break
                else:
                    print(f"{agent_name}: ✗ No solution found")
            
            # Manual step-by-step analysis for level 1 (simplest rule formation case)
            if level_id == '1':
                print(f"\nMANUAL ANALYSIS FOR LEVEL 1:")
                test_state = initial_state.copy()
                
                # Try to manually form flag-is-win rule
                print(f"Looking for flag-is-win formation...")
                
                # Find word positions
                flag_words = [w for w in test_state.words if w.name == "flag"]
                is_words = [w for w in test_state.words if w.name == "is"]
                win_words = [w for w in test_state.words if w.name == "win"]
                
                print(f"Flag words at: {[(w.x, w.y) for w in flag_words]}")
                print(f"Is words at: {[(w.x, w.y) for w in is_words]}")
                print(f"Win words at: {[(w.x, w.y) for w in win_words]}")
                
                # Test a few moves from the expected solution
                print(f"Testing first few moves from expected solution...")
                for i, char in enumerate(expected_solution[:6]):
                    direction = Direction(char.lower())
                    if direction != Direction.Undefined:
                        test_state = advance_game_state(direction, test_state)
                        print(f"After move {i+1} ({char}): Rules = {test_state.rules}")
                        if "flag-is-win" in test_state.rules:
                            print(f"★ FLAG-IS-WIN rule formed after move {i+1}!")
                            break

def test_specific_rule_formation():
    """Test the rule formation cost calculation"""
    print(f"\n" + "="*60)
    print("TESTING RULE FORMATION CALCULATIONS")
    
    # Load level 1 for testing
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
    level1 = next(l for l in data['levels'] if l['id'] == '1')
    
    map_data = parse_map(level1['ascii'])
    initial_state = make_level(map_data)
    
    agent = EnhancedASTARAgent()
    
    # Test rule formation costs
    flag_win_cost = agent._estimate_rule_formation_cost(initial_state, "flag", "is", "win")
    baba_you_cost = agent._estimate_rule_formation_cost(initial_state, "baba", "is", "you")
    
    print(f"Flag-is-win formation cost: {flag_win_cost}")
    print(f"Baba-is-you formation cost: {baba_you_cost}")
    
    initial_heuristic = agent._calculate_heuristic(initial_state)
    print(f"Initial heuristic: {initial_heuristic}")

if __name__ == "__main__":
    test_win_rule_formation()
    test_specific_rule_formation()
