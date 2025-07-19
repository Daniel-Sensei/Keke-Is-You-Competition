"""
Test to determine if the hybrid agent's failures are due to classification issues
or inability to find solutions.
"""

import json
import time
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from baba import GameState, Direction, make_level, advance_game_state, check_win, parse_map
from agents.improved_evolutionary_AGENT import IMPROVED_EVOLUTIONARYAgent
from agents.hybrid_AGENT import HYBRIDAgent

def test_agent_on_levels(agent, agent_name, levels, max_levels=5):
    """Test an agent on specific levels and return results."""
    results = {}
    
    for i, level_data in enumerate(levels[:max_levels]):
        print(f"\n{'='*60}")
        print(f"Testing {agent_name} on Level {i+1}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            # Convert level data to GameState
            ascii_map = level_data['ascii']
            parsed_map = parse_map(ascii_map)
            level = make_level(parsed_map)
            
            # Test the agent directly
            solution = agent.search(level, iterations=100)
            
            elapsed_time = time.time() - start_time
            
            # Check if solution actually solves the level
            solved = False
            if solution:
                try:
                    current_state = level
                    for action in solution:
                        current_state = advance_game_state(action, current_state.copy())
                        if check_win(current_state):
                            solved = True
                            break
                except Exception as e:
                    print(f"Error validating solution: {e}")
                    solved = False
            
            results[f"level_{i+1}"] = {
                'solved': solved,
                'solution_length': len(solution) if solution else 0,
                'time': elapsed_time,
                'solution': solution[:20] if solution else []  # First 20 moves only
            }
            
            print(f"Result: {'SOLVED' if solved else 'FAILED'}")
            print(f"Solution length: {len(solution) if solution else 0}")
            print(f"Time: {elapsed_time:.2f}s")
            
            if solution:
                print(f"First 20 moves: {solution[:20]}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[f"level_{i+1}"] = {
                'solved': False,
                'solution_length': 0,
                'time': elapsed_time,
                'error': str(e),
                'solution': []
            }
            print(f"Result: ERROR - {e}")
    
    return results

def main():
    # Load the first few levels from train_LEVELS.json
    with open('json_levels/train_LEVELS.json', 'r') as f:
        data = json.load(f)
        train_levels = data.get('levels', [])
    
    print("Loaded train levels:", len(train_levels))
    
    # Test the improved evolutionary agent (with classification)
    print("\n" + "="*80)
    print("TESTING IMPROVED EVOLUTIONARY AGENT (WITH CLASSIFICATION)")
    print("="*80)
    
    improved_evo_agent = IMPROVED_EVOLUTIONARYAgent(
        population_size=50, 
        generations=100, 
        solution_length=80
    )
    
    improved_results = test_agent_on_levels(
        improved_evo_agent, 
        "Improved Evolutionary Agent", 
        train_levels, 
        max_levels=5
    )
    
    # Test the hybrid agent (no classification)
    print("\n" + "="*80)
    print("TESTING HYBRID AGENT (NO CLASSIFICATION)")
    print("="*80)
    
    hybrid_agent = HYBRIDAgent(max_time_per_method=30.0, switch_threshold=50)
    
    hybrid_results = test_agent_on_levels(
        hybrid_agent, 
        "Hybrid Agent", 
        train_levels, 
        max_levels=5
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Level':<10} {'Improved Evo':<15} {'Hybrid':<15} {'Analysis':<30}")
    print("-" * 70)
    
    for i in range(1, 6):
        level_key = f"level_{i}"
        
        improved_solved = improved_results.get(level_key, {}).get('solved', False)
        hybrid_solved = hybrid_results.get(level_key, {}).get('solved', False)
        
        improved_status = "SOLVED" if improved_solved else "FAILED"
        hybrid_status = "SOLVED" if hybrid_solved else "FAILED"
        
        # Analysis
        if improved_solved and hybrid_solved:
            analysis = "Both solved"
        elif not improved_solved and hybrid_solved:
            analysis = "Classification issue"
        elif improved_solved and not hybrid_solved:
            analysis = "Search issue"
        else:
            analysis = "Both failed"
        
        print(f"{i:<10} {improved_status:<15} {hybrid_status:<15} {analysis:<30}")
    
    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    print("\nImproved Evolutionary Agent Results:")
    for level, result in improved_results.items():
        print(f"  {level}: {result}")
    
    print("\nHybrid Agent Results:")
    for level, result in hybrid_results.items():
        print(f"  {level}: {result}")

if __name__ == "__main__":
    main()
