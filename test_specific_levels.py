#!/usr/bin/env python3
"""
Test specific levels that are failing with the improved agent
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from execution import Execution

def run_agent_on_specific_levels(agent_path, levels_data, iterations=100):
    """
    Run an agent on specific level data and return results
    """
    # Save levels to temporary file
    temp_levels_file = f"temp_{agent_path.replace('/', '_').replace('.py', '')}_levels.json"
    with open(temp_levels_file, 'w') as f:
        json.dump(levels_data, f)
    
    try:
        # Create execution instance
        exec_instance = Execution(agent_path, temp_levels_file, iter_cap=iterations, use_cache=False)
        
        # Run each level and collect results
        results = []
        for level_data in exec_instance.levels:
            start_time = time.time()
            result = exec_instance.run_single_level(level_data)
            exec_time = time.time() - start_time
            
            # Format result consistently
            formatted_result = {
                'success': result['won_level'],
                'moves': len(result['solution']) if result['solution'] else 0,
                'time': exec_time,
                'solution': result['solution'] if result['solution'] else [],
                'level_id': level_data.get('id', 'unknown')
            }
            results.append(formatted_result)
            
        return results
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_levels_file):
            os.remove(temp_levels_file)
def test_specific_levels():
    """Test the first two levels from train_LEVELS.json that are failing"""
    
    # Define the specific levels that are failing
    failing_levels = {
        "levels": [
            {
                "id": "1",
                "name": None,
                "author": "PCG.js",
                "ascii": "______\n_K.2._\n_kSF._\n_bB1._\n_..fR_\n_1..._\n_3.r._\n_251O_\n_s.31_\n_o..2_\n______",
                "solution": "RRUUULUULURL"
            },
            {
                "id": "6", 
                "name": None,
                "author": "Baba",
                "ascii": "________\n_......_\n_......_\n_.Sb.s._\n_.1..s._\n_.4b..1_\n_B12..3_\n_......_\n________",
                "solution": "UUULLDRRRRURDL"
            }
        ]
    }
    
    print("=== TESTING SPECIFIC FAILING LEVELS ===")
    print(f"Testing {len(failing_levels['levels'])} levels that are failing...")
    
    # Test original agent
    print("\n--- Original Agent ---")
    original_results = run_agent_on_specific_levels(
        agent_path="agents/evolutionary_AGENT.py",
        levels_data=failing_levels,
        iterations=100
    )
    
    # Test improved agent
    print("\n--- Improved Agent ---")
    improved_results = run_agent_on_specific_levels(
        agent_path="agents/improved_evolutionary_AGENT.py", 
        levels_data=failing_levels,
        iterations=100
    )
    
    # Analyze results
    print("\n=== ANALYSIS ===")
    for i, level in enumerate(failing_levels['levels']):
        level_id = level['id']
        expected_solution = level['solution']
        
        print(f"\nLevel {level_id}:")
        print(f"Expected solution: {expected_solution} ({len(expected_solution)} moves)")
        print(f"ASCII map:")
        for line in level['ascii'].split('\\n'):
            print(f"  {line}")
        
        if i < len(original_results):
            orig_result = original_results[i]
            print(f"Original: {'✅' if orig_result['success'] else '❌'} - {orig_result.get('moves', 0)} moves, {orig_result.get('time', 0):.2f}s")
            if orig_result['success'] and 'solution' in orig_result:
                print(f"  Solution: {''.join(orig_result['solution'])}")
        
        if i < len(improved_results):
            imp_result = improved_results[i] 
            print(f"Improved: {'✅' if imp_result['success'] else '❌'} - {imp_result.get('moves', 0)} moves, {imp_result.get('time', 0):.2f}s")
            if imp_result['success'] and 'solution' in imp_result:
                print(f"  Solution: {''.join(imp_result['solution'])}")

if __name__ == "__main__":
    test_specific_levels()
