#!/usr/bin/env python3
"""
Test script to verify the improved evolutionary agent respects iteration limits
and produces optimized solutions.
"""

from execution import Execution

def test_improved_evolutionary():
    print("Testing improved evolutionary agent...")
    
    # Test with limited iterations
    exec_instance = Execution(
        'agents/improved_evolutionary_AGENT.py', 
        'json_levels/test_LEVELS.json', 
        iter_cap=30
    )
    
    # Run first level
    print("Running first level with 30 iteration limit...")
    if exec_instance.levels and len(exec_instance.levels) > 0:
        level_data = exec_instance.levels[0]
        result = exec_instance.run_single_level(level_data)
        
        print(f"Result:")
        print(f"  - Won: {result['won_level']}")
        print(f"  - Solution length: {len(result['solution']) if result['solution'] else 0} moves")
        print(f"  - Time: {result['time']:.2f}s")
        print(f"  - Iterations used: {result['iterations']}")
        
        if result['solution']:
            print(f"  - First 10 moves: {result['solution'][:10]}")
    else:
        print("No levels loaded!")

if __name__ == "__main__":
    test_improved_evolutionary()
