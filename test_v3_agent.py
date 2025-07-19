#!/usr/bin/env python3

import json
import time
from agents.enhanced_astar_v3_AGENT import ENHANCED_ASTAR_V3_Agent
from baba import GameState, make_level, parse_map

def test_v3_agent():
    """Testa l'agente v3 sui livelli problematici."""
    
    # Carica i livelli
    with open('json_levels/train_LEVELS.json', 'r') as f:
        train_data = json.load(f)
    
    with open('json_levels/test_LEVELS.json', 'r') as f:
        test_data = json.load(f)
    
    # Combina i livelli
    all_levels = train_data['levels'] + test_data['levels']
    
    # Trova i livelli problematici
    problematic_ids = ['72', '15', '70']  # Inizia con 72 che è più corto
    problematic_levels = []
    
    for level in all_levels:
        if level['id'] in problematic_ids:
            problematic_levels.append(level)
    
    # Ordina per lunghezza soluzione attesa
    problematic_levels.sort(key=lambda x: len(x['solution']))
    
    print(f"Test Enhanced A* V3 su {len(problematic_levels)} livelli problematici")
    
    # Inizializza l'agente
    agent = ENHANCED_ASTAR_V3_Agent()
    
    results = []
    
    # Testa ogni livello problematico
    for level in problematic_levels:
        level_id = level['id']
        print(f"\n{'='*60}")
        print(f"Testing Level {level_id}: {level.get('name', 'Unnamed')}")
        print(f"Expected solution length: {len(level['solution'])}")
        
        # Decodifica il game state
        game_map = parse_map(level['ascii'])
        game_state = make_level(game_map)
        
        # Analisi rapida
        print(f"Players: {len(game_state.players)}, Winnables: {len(game_state.winnables)}")
        
        # Testa l'agente
        start_time = time.time()
        
        try:
            solution = agent.search(game_state.copy())
            elapsed = time.time() - start_time
            
            if solution:
                valid = agent._validate_solution(game_state, solution)
                results.append({
                    'level': level_id,
                    'success': True,
                    'moves': len(solution),
                    'expected_moves': len(level['solution']),
                    'time': elapsed,
                    'valid': valid,
                    'solution': ''.join([d.name[0] for d in solution])
                })
                print(f"✓ V3: {len(solution)} moves in {elapsed:.2f}s, valid: {valid}")
                print(f"✓ Expected: {len(level['solution'])} moves")
                print(f"✓ Solution: {results[-1]['solution']}")
            else:
                results.append({
                    'level': level_id,
                    'success': False,
                    'time': elapsed
                })
                print(f"✗ V3: No solution in {elapsed:.2f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'level': level_id,
                'success': False,
                'time': elapsed,
                'error': str(e)
            })
            print(f"✗ V3: Error after {elapsed:.2f}s: {e}")
    
    # Stampa riepilogo
    print(f"\n{'='*60}")
    print("RIEPILOGO RISULTATI V3:")
    print(f"{'Level':<8} {'Success':<8} {'Moves':<8} {'Expected':<8} {'Time':<8} {'Valid':<8}")
    print("-" * 60)
    
    successes = 0
    for result in results:
        success = result['success']
        moves = result.get('moves', 0)
        expected = result.get('expected_moves', 0)
        time_taken = result['time']
        valid = result.get('valid', False)
        
        if success:
            successes += 1
            print(f"{result['level']:<8} {'✓':<8} {moves:<8} {expected:<8} {time_taken:<8.2f} {str(valid):<8}")
        else:
            print(f"{result['level']:<8} {'✗':<8} {'-':<8} {expected:<8} {time_taken:<8.2f} {'-':<8}")
    
    print(f"\nSuccesso: {successes}/{len(results)} livelli risolti")
    
    if successes > 0:
        print("🎉 Enhanced A* V3 ha risolto alcuni livelli problematici!")
        return True
    else:
        print("😞 Enhanced A* V3 non ha risolto nessun livello problematico.")
        return False

if __name__ == "__main__":
    test_v3_agent()
