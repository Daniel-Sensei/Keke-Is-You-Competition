#!/usr/bin/env python3

import json
import time
from agents.enhanced_astar_AGENT import ENHANCED_ASTARAgent
from baba import GameState, make_level, parse_map

def test_problematic_levels():
    """Test dei livelli problematici: 15, 70, 71, 72"""
    
    # Carica i livelli
    with open('json_levels/train_LEVELS.json', 'r') as f:
        train_data = json.load(f)
    
    with open('json_levels/test_LEVELS.json', 'r') as f:
        test_data = json.load(f)
    
    # Combina i livelli
    all_levels = train_data['levels'] + test_data['levels']
    
    # Trova i livelli problematici
    problematic_ids = ['15', '70', '71', '72']
    problematic_levels = []
    
    for level in all_levels:
        if level['id'] in problematic_ids:
            problematic_levels.append(level)
    
    print(f"Trovati {len(problematic_levels)} livelli problematici")
    
    # Inizializza l'agente
    agent = ENHANCED_ASTARAgent()
    
    # Testa ogni livello problematico
    for level in problematic_levels:
        print(f"\n{'='*60}")
        print(f"Testing Level {level['id']}: {level.get('name', 'Unnamed')}")
        print(f"Author: {level['author']}")
        print(f"ASCII:")
        print(level['ascii'])
        print(f"Expected solution length: {len(level['solution'])}")
        print(f"Expected solution: {level['solution']}")
        
        # Decodifica il game state
        game_map = parse_map(level['ascii'])
        game_state = make_level(game_map)
        
        # Analizza le caratteristiche del livello
        print(f"\nLevel analysis:")
        print(f"- Players: {len(game_state.players)}")
        print(f"- Winnables: {len(game_state.winnables)}")
        print(f"- Killers: {len(game_state.killers)}")
        print(f"- Sinkers: {len(game_state.sinkers)}")
        print(f"- Pushables: {len(game_state.pushables)}")
        print(f"- Physical objects: {len(game_state.phys)}")
        print(f"- Rules: {len(game_state.rules)}")
        print(f"- Auto movers: {len(game_state.auto_movers)}")
        
        # Stampa le regole attive
        if game_state.rules:
            print(f"- Active rules: {game_state.rules[:10]}...")  # Prime 10 regole
        
        # Testa l'agente
        print(f"\nTesting agent...")
        start_time = time.time()
        
        try:
            solution = agent.search(game_state)
            elapsed = time.time() - start_time
            
            if solution:
                print(f"✓ Solution found in {elapsed:.2f}s: {len(solution)} moves")
                print(f"Solution: {''.join([d.name[0] for d in solution])}")
                
                # Verifica la soluzione
                if agent._validate_solution(game_state, solution):
                    print("✓ Solution validated successfully!")
                else:
                    print("✗ Solution validation failed!")
            else:
                print(f"✗ No solution found in {elapsed:.2f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Error after {elapsed:.2f}s: {e}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")

if __name__ == "__main__":
    test_problematic_levels()
