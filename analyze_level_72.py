#!/usr/bin/env python3

import json
from baba import make_level, parse_map, check_win, advance_game_state, Direction

def analyze_level_72():
    """Analizza in dettaglio il livello 72."""
    
    # Carica il livello 72
    with open('json_levels/train_LEVELS.json', 'r') as f:
        train_data = json.load(f)
    
    level_72 = None
    for level in train_data['levels']:
        if level['id'] == '72':
            level_72 = level
            break
    
    if not level_72:
        print("Livello 72 non trovato!")
        return
    
    print("Livello 72 - Analisi dettagliata:")
    print(f"ASCII:\n{level_72['ascii']}")
    print(f"Soluzione attesa: {level_72['solution']}")
    print(f"Lunghezza soluzione: {len(level_72['solution'])}")
    
    # Decodifica il game state
    game_map = parse_map(level_72['ascii'])
    game_state = make_level(game_map)
    
    print(f"\nStato iniziale:")
    print(f"- Players: {len(game_state.players)}")
    print(f"- Winnables: {len(game_state.winnables)}")
    print(f"- Regole: {game_state.rules}")
    
    # Prova a seguire la soluzione passo per passo
    print(f"\nSimulazione della soluzione:")
    current_state = game_state
    solution_moves = level_72['solution']
    
    direction_map = {
        'L': Direction.Left,
        'R': Direction.Right,
        'U': Direction.Up,
        'D': Direction.Down
    }
    
    for i, move_char in enumerate(solution_moves):
        if move_char.upper() in direction_map:
            direction = direction_map[move_char.upper()]
            print(f"Mossa {i+1}: {move_char} -> {direction}")
            
            # Avanza lo stato
            new_state = advance_game_state(direction, current_state.copy())
            
            # Controlla se abbiamo vinto
            if check_win(new_state):
                print(f"  -> VITTORIA alla mossa {i+1}!")
                print(f"  -> Winnables dopo la mossa: {len(new_state.winnables)}")
                
                # Mostra come sono cambiati i winnables
                if new_state.winnables:
                    print(f"  -> Oggetti vincenti:")
                    for w in new_state.winnables:
                        print(f"     - {w.name} at ({w.x}, {w.y})")
                break
            
            current_state = new_state
            
            # Mostra lo stato dopo alcune mosse chiave
            if i < 5 or i % 5 == 0:
                print(f"  -> Players: {len(current_state.players)}, Winnables: {len(current_state.winnables)}")
                print(f"  -> Regole: {current_state.rules}")
    
    print(f"\nAnalisi completata!")

if __name__ == "__main__":
    analyze_level_72()
