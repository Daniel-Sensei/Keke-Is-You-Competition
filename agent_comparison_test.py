#!/usr/bin/env python3
"""
Comparison test between original evolutionary and improved evolutionary agents.
"""

from execution import Execution
import time

def test_agent_comparison():
    print("=== AGENT COMPARISON TEST ===\n")
    
    # Test parameters
    test_iterations = 50
    level_index = 0  # First level
    
    # Test original evolutionary agent
    print("1. Testing ORIGINAL Evolutionary Agent...")
    start_time = time.time()
    
    exec_original = Execution(
        'agents/evolutionary_AGENT.py', 
        'json_levels/test_LEVELS.json', 
        iter_cap=test_iterations
    )
    
    if exec_original.levels and len(exec_original.levels) > level_index:
        level_data = exec_original.levels[level_index]
        result_original = exec_original.run_single_level(level_data)
        original_time = time.time() - start_time
        
        print(f"   - Won: {result_original['won_level']}")
        print(f"   - Solution length: {len(result_original['solution']) if result_original['solution'] else 0} moves")
        print(f"   - Time: {original_time:.3f}s")
        print(f"   - Iterations used: {result_original['iterations']}")
    else:
        print("   - ERROR: No levels found")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Test improved evolutionary agent
    print("2. Testing IMPROVED Evolutionary Agent...")
    start_time = time.time()
    
    exec_improved = Execution(
        'agents/improved_evolutionary_AGENT.py', 
        'json_levels/test_LEVELS.json', 
        iter_cap=test_iterations
    )
    
    if exec_improved.levels and len(exec_improved.levels) > level_index:
        level_data = exec_improved.levels[level_index]
        result_improved = exec_improved.run_single_level(level_data)
        improved_time = time.time() - start_time
        
        print(f"   - Won: {result_improved['won_level']}")
        print(f"   - Solution length: {len(result_improved['solution']) if result_improved['solution'] else 0} moves")
        print(f"   - Time: {improved_time:.3f}s")
        print(f"   - Iterations used: {result_improved['iterations']}")
    else:
        print("   - ERROR: No levels found")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Comparison summary
    print("3. COMPARISON SUMMARY:")
    
    if result_original['won_level'] and result_improved['won_level']:
        original_moves = len(result_original['solution']) if result_original['solution'] else 0
        improved_moves = len(result_improved['solution']) if result_improved['solution'] else 0
        
        if original_moves > 0 and improved_moves > 0:
            efficiency_improvement = ((original_moves - improved_moves) / original_moves) * 100
            time_ratio = improved_time / original_time if original_time > 0 else 1
            
            print(f"   - Solution efficiency: {efficiency_improvement:+.1f}% ({original_moves} -> {improved_moves} moves)")
            print(f"   - Time ratio: {time_ratio:.2f}x ({'faster' if time_ratio < 1 else 'slower'})")
            print(f"   - Both agents: {'✅ SOLVED' if result_original['won_level'] and result_improved['won_level'] else '❌ FAILED'}")
            
            if improved_moves <= original_moves and improved_time <= original_time * 1.5:
                print("   - Result: ✅ IMPROVED agent is better or comparable")
            elif improved_moves > original_moves * 1.5:
                print("   - Result: ⚠️  IMPROVED agent uses too many moves")
            else:
                print("   - Result: ⚠️  Mixed results")
        else:
            print("   - Result: ❌ One or both agents failed to find solutions")
    else:
        print("   - Result: ❌ One or both agents failed to solve the level")

if __name__ == "__main__":
    test_agent_comparison()
