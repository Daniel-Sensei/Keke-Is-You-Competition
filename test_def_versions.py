#!/usr/bin/env python3

import json
import time
from pathlib import Path
import sys
import os

# Add the current directory to Python path to import agents
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.def_AGENT import DEFAgent
from agents.defv1_AGENT import DEFAgentV1
from agents.defv2_AGENT import DEFAgentV2
from agents.defV3_AGENT import DEFAgentV3
from agents.defv4_AGENT import DEFAgentV4
from baba import GameState, Direction
from execution import run_on_level


def test_agents_comparison():
    """Test different versions of DEF agent on a few levels to identify problems"""
    
    agents = {
        'Original DEF': DEFAgent(),
        'V1 (Baseline)': DEFAgentV1(),
        'V2 (+Cache)': DEFAgentV2(),
        'V3 (+Reduced Iter)': DEFAgentV3(),
        'V4 (+Light Pruning)': DEFAgentV4()
    }
    
    # Test on some easy levels first
    test_levels = [1, 2, 3, 5, 10]  # Start with easier levels
    
    results = {}
    
    print("Testing DEF Agent versions...")
    print("=" * 60)
    
    for level_num in test_levels:
        print(f"\nLevel {level_num}:")
        print("-" * 30)
        
        level_results = {}
        
        for agent_name, agent in agents.items():
            print(f"Testing {agent_name}...", end=" ")
            
            try:
                start_time = time.time()
                path, iterations = run_on_level(level_num, agent, max_iterations=10000)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                if path is not None:
                    status = "SOLVED"
                    path_length = len(path)
                    print(f"✓ {status} in {execution_time:.2f}s (path: {path_length}, iter: {iterations})")
                else:
                    status = "FAILED"
                    path_length = 0
                    print(f"✗ {status} in {execution_time:.2f}s (iter: {iterations})")
                
                level_results[agent_name] = {
                    'status': status,
                    'time': execution_time,
                    'path_length': path_length,
                    'iterations': iterations
                }
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                level_results[agent_name] = {
                    'status': 'ERROR',
                    'time': 0,
                    'path_length': 0,
                    'iterations': 0,
                    'error': str(e)
                }
        
        results[level_num] = level_results
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    # Create summary table
    for agent_name in agents.keys():
        solved_count = 0
        total_time = 0
        avg_path_length = 0
        valid_solutions = 0
        
        for level_num in test_levels:
            if level_num in results and agent_name in results[level_num]:
                result = results[level_num][agent_name]
                if result['status'] == 'SOLVED':
                    solved_count += 1
                    total_time += result['time']
                    avg_path_length += result['path_length']
                    valid_solutions += 1
        
        if valid_solutions > 0:
            avg_path_length = avg_path_length / valid_solutions
            avg_time = total_time / valid_solutions
        else:
            avg_path_length = 0
            avg_time = 0
        
        print(f"\n{agent_name}:")
        print(f"  Solved: {solved_count}/{len(test_levels)}")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Avg path length: {avg_path_length:.1f}")
    
    # Identify problematic changes
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    original_solved = set()
    for level_num in test_levels:
        if (level_num in results and 
            'Original DEF' in results[level_num] and 
            results[level_num]['Original DEF']['status'] == 'SOLVED'):
            original_solved.add(level_num)
    
    print(f"Original DEF solved: {sorted(original_solved)}")
    
    for agent_name in ['V1 (Baseline)', 'V2 (+Cache)', 'V3 (+Reduced Iter)', 'V4 (+Light Pruning)']:
        agent_solved = set()
        for level_num in test_levels:
            if (level_num in results and 
                agent_name in results[level_num] and 
                results[level_num][agent_name]['status'] == 'SOLVED'):
                agent_solved.add(level_num)
        
        lost_levels = original_solved - agent_solved
        gained_levels = agent_solved - original_solved
        
        print(f"\n{agent_name}:")
        print(f"  Solved: {sorted(agent_solved)}")
        if lost_levels:
            print(f"  ⚠️  Lost levels: {sorted(lost_levels)}")
        if gained_levels:
            print(f"  ✓ Gained levels: {sorted(gained_levels)}")
    
    return results


if __name__ == "__main__":
    test_agents_comparison()
