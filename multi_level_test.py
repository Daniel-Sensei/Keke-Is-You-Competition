#!/usr/bin/env python3
"""
Multi-level comparison test between evolutionary agents.
"""

from execution import Execution
import time

def test_multiple_levels():
    print("=== MULTI-LEVEL AGENT COMPARISON ===\n")
    
    # Test parameters
    test_iterations = 100
    num_levels_to_test = 10  # Test first 10 levels
    
    results = {
        'original': {'solved': 0, 'total_time': 0, 'total_moves': 0, 'solutions': []},
        'improved': {'solved': 0, 'total_time': 0, 'total_moves': 0, 'solutions': []}
    }
    
    # Load levels once
    exec_original = Execution('agents/evolutionary_AGENT.py', 'json_levels/test_LEVELS.json', iter_cap=test_iterations)
    exec_improved = Execution('agents/improved_evolutionary_AGENT.py', 'json_levels/test_LEVELS.json', iter_cap=test_iterations)
    
    if not exec_original.levels or len(exec_original.levels) < num_levels_to_test:
        print("ERROR: Not enough levels loaded")
        return
    
    failed_levels = []
    
    for level_idx in range(num_levels_to_test):
        print(f"\n--- Testing Level {level_idx + 1}/{num_levels_to_test} ---")
        level_data = exec_original.levels[level_idx]
        level_name = level_data.get('name', f'Level_{level_idx}')
        print(f"Level: {level_name}")
        
        # Test original evolutionary
        print(f"  Original: ", end="", flush=True)
        start_time = time.time()
        result_orig = exec_original.run_single_level(level_data)
        orig_time = time.time() - start_time
        
        if result_orig['won_level']:
            results['original']['solved'] += 1
            results['original']['total_time'] += orig_time
            orig_moves = len(result_orig['solution']) if result_orig['solution'] else 0
            results['original']['total_moves'] += orig_moves
            results['original']['solutions'].append(orig_moves)
            print(f"✅ {orig_moves} moves, {orig_time:.2f}s")
        else:
            results['original']['solutions'].append(0)
            print(f"❌ Failed, {orig_time:.2f}s")
        
        # Test improved evolutionary
        print(f"  Improved: ", end="", flush=True)
        start_time = time.time()
        result_imp = exec_improved.run_single_level(level_data)
        imp_time = time.time() - start_time
        
        if result_imp['won_level']:
            results['improved']['solved'] += 1
            results['improved']['total_time'] += imp_time
            imp_moves = len(result_imp['solution']) if result_imp['solution'] else 0
            results['improved']['total_moves'] += imp_moves
            results['improved']['solutions'].append(imp_moves)
            print(f"✅ {imp_moves} moves, {imp_time:.2f}s")
        else:
            results['improved']['solutions'].append(0)
            print(f"❌ Failed, {imp_time:.2f}s")
            
            # Track levels where improved failed but original succeeded
            if result_orig['won_level']:
                failed_levels.append({
                    'index': level_idx,
                    'name': level_name,
                    'original_moves': len(result_orig['solution']) if result_orig['solution'] else 0,
                    'ascii': level_data.get('ascii', 'N/A')
                })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON:")
    print("="*60)
    
    print(f"Levels solved:")
    print(f"  Original:  {results['original']['solved']}/{num_levels_to_test} ({results['original']['solved']/num_levels_to_test*100:.1f}%)")
    print(f"  Improved:  {results['improved']['solved']}/{num_levels_to_test} ({results['improved']['solved']/num_levels_to_test*100:.1f}%)")
    
    if results['original']['solved'] > 0:
        avg_time_orig = results['original']['total_time'] / results['original']['solved']
        avg_moves_orig = results['original']['total_moves'] / results['original']['solved']
        print(f"  Original avg: {avg_moves_orig:.1f} moves, {avg_time_orig:.2f}s per level")
    
    if results['improved']['solved'] > 0:
        avg_time_imp = results['improved']['total_time'] / results['improved']['solved']
        avg_moves_imp = results['improved']['total_moves'] / results['improved']['solved']
        print(f"  Improved avg: {avg_moves_imp:.1f} moves, {avg_time_imp:.2f}s per level")
    
    # Failed levels analysis
    if failed_levels:
        print(f"\n⚠️  LEVELS WHERE IMPROVED FAILED BUT ORIGINAL SUCCEEDED:")
        print("="*60)
        for fail in failed_levels:
            print(f"Level {fail['index'] + 1}: {fail['name']} (Original: {fail['original_moves']} moves)")
            print(f"ASCII Map:")
            print(fail['ascii'][:200] + ("..." if len(fail['ascii']) > 200 else ""))
            print("-" * 40)
    
    # Performance comparison
    if results['improved']['solved'] >= results['original']['solved'] * 0.8:  # At least 80% success rate
        print(f"\n✅ IMPROVED agent performance: ACCEPTABLE")
    else:
        print(f"\n⚠️  IMPROVED agent performance: NEEDS IMPROVEMENT")
        print(f"   Success rate dropped from {results['original']['solved']} to {results['improved']['solved']} levels")

if __name__ == "__main__":
    test_multiple_levels()
