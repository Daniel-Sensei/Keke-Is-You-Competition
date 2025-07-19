#!/usr/bin/env python3

import json
import time
from agents.enhanced_astar_AGENT import ENHANCED_ASTARAgent
from agents.enhanced_astar_v2_AGENT import ENHANCED_ASTAR_V2_Agent
from baba import GameState, make_level, parse_map

def test_agent_comparison():
    """Confronta le due versioni dell'agente sui livelli problematici."""
    
    # Carica i livelli
    with open('json_levels/train_LEVELS.json', 'r') as f:
        train_data = json.load(f)
    
    with open('json_levels/test_LEVELS.json', 'r') as f:
        test_data = json.load(f)
    
    # Combina i livelli
    all_levels = train_data['levels'] + test_data['levels']
    
    # Trova i livelli problematici
    problematic_ids = ['15', '70', '72']  # Rimuovo 71 perché già funziona
    problematic_levels = []
    
    for level in all_levels:
        if level['id'] in problematic_ids:
            problematic_levels.append(level)
    
    print(f"Confronto su {len(problematic_levels)} livelli problematici")
    
    # Inizializza gli agenti
    agent_v1 = ENHANCED_ASTARAgent()
    agent_v2 = ENHANCED_ASTAR_V2_Agent()
    
    results = {}
    
    # Testa ogni livello problematico
    for level in problematic_levels:
        level_id = level['id']
        print(f"\n{'='*60}")
        print(f"Testing Level {level_id}: {level.get('name', 'Unnamed')}")
        print(f"Expected solution length: {len(level['solution'])}")
        
        # Decodifica il game state
        game_map = parse_map(level['ascii'])
        game_state = make_level(game_map)
        
        results[level_id] = {}
        
        # Test Agent V1
        print(f"\n--- Testing Enhanced A* V1 ---")
        start_time = time.time()
        
        try:
            solution_v1 = agent_v1.search(game_state.copy())
            elapsed_v1 = time.time() - start_time
            
            if solution_v1:
                valid_v1 = agent_v1._validate_solution(game_state, solution_v1)
                results[level_id]['v1'] = {
                    'success': True,
                    'moves': len(solution_v1),
                    'time': elapsed_v1,
                    'valid': valid_v1,
                    'solution': ''.join([d.name[0] for d in solution_v1])
                }
                print(f"✓ V1: {len(solution_v1)} moves in {elapsed_v1:.2f}s, valid: {valid_v1}")
            else:
                results[level_id]['v1'] = {
                    'success': False,
                    'time': elapsed_v1
                }
                print(f"✗ V1: No solution in {elapsed_v1:.2f}s")
                
        except Exception as e:
            elapsed_v1 = time.time() - start_time
            results[level_id]['v1'] = {
                'success': False,
                'time': elapsed_v1,
                'error': str(e)
            }
            print(f"✗ V1: Error after {elapsed_v1:.2f}s: {e}")
        
        # Test Agent V2
        print(f"\n--- Testing Enhanced A* V2 ---")
        start_time = time.time()
        
        try:
            solution_v2 = agent_v2.search(game_state.copy())
            elapsed_v2 = time.time() - start_time
            
            if solution_v2:
                valid_v2 = agent_v2._validate_solution(game_state, solution_v2)
                results[level_id]['v2'] = {
                    'success': True,
                    'moves': len(solution_v2),
                    'time': elapsed_v2,
                    'valid': valid_v2,
                    'solution': ''.join([d.name[0] for d in solution_v2])
                }
                print(f"✓ V2: {len(solution_v2)} moves in {elapsed_v2:.2f}s, valid: {valid_v2}")
            else:
                results[level_id]['v2'] = {
                    'success': False,
                    'time': elapsed_v2
                }
                print(f"✗ V2: No solution in {elapsed_v2:.2f}s")
                
        except Exception as e:
            elapsed_v2 = time.time() - start_time
            results[level_id]['v2'] = {
                'success': False,
                'time': elapsed_v2,
                'error': str(e)
            }
            print(f"✗ V2: Error after {elapsed_v2:.2f}s: {e}")
    
    # Stampa riepilogo
    print(f"\n{'='*60}")
    print("RIEPILOGO RISULTATI:")
    print(f"{'Level':<8} {'V1 Success':<12} {'V1 Time':<10} {'V2 Success':<12} {'V2 Time':<10} {'Winner':<10}")
    print("-" * 60)
    
    v1_wins = 0
    v2_wins = 0
    
    for level_id, result in results.items():
        v1_success = result['v1']['success'] if 'v1' in result else False
        v2_success = result['v2']['success'] if 'v2' in result else False
        
        v1_time = result['v1']['time'] if 'v1' in result else 0
        v2_time = result['v2']['time'] if 'v2' in result else 0
        
        if v1_success and v2_success:
            # Entrambi hanno successo, vince chi è più veloce
            winner = "V1" if v1_time < v2_time else "V2"
            if winner == "V1":
                v1_wins += 1
            else:
                v2_wins += 1
        elif v1_success:
            winner = "V1"
            v1_wins += 1
        elif v2_success:
            winner = "V2"
            v2_wins += 1
        else:
            winner = "NONE"
        
        print(f"{level_id:<8} {str(v1_success):<12} {v1_time:<10.2f} {str(v2_success):<12} {v2_time:<10.2f} {winner:<10}")
    
    print(f"\nRisultato finale: V1 = {v1_wins}, V2 = {v2_wins}")
    
    if v2_wins > v1_wins:
        print("🎉 Enhanced A* V2 ha prestazioni migliori!")
    elif v1_wins > v2_wins:
        print("🤔 Enhanced A* V1 ha prestazioni migliori!")
    else:
        print("🤝 Prestazioni equivalenti!")

if __name__ == "__main__":
    test_agent_comparison()
