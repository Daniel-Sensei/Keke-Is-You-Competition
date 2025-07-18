# find_contradictions.py

import json
import numpy as np
import os
from collections import defaultdict

from baba_env import BabaEnv

def find_state_collisions(levels: list, max_h: int, max_w: int):
    """
    Analizza il dataset generato per trovare stati identici che richiedono azioni diverse.
    """
    print("🔬 Inizio analisi del dataset per trovare collisioni di stati...")
    
    # Usiamo un dizionario per mappare l'hash di uno stato alla sua azione e provenienza
    state_map = {}
    collisions_found = 0
    
    move_to_action_idx = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    action_idx_to_move = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for level_idx, level in enumerate(levels):
        env = BabaEnv([level['ascii']], max_height=max_h, max_width=max_w)
        obs, _ = env.reset()
        
        for step_idx, move_char in enumerate(level['solution'].lower()):
            if move_char not in move_to_action_idx: continue
            
            expert_action = move_to_action_idx[move_char]
            
            # Converte l'array numpy in una tupla di byte per renderlo hashable (usabile come chiave di dizionario)
            state_key = obs.tobytes()

            if state_key not in state_map:
                # Se è la prima volta che vediamo questo stato, lo salviamo
                state_map[state_key] = {
                    'action': expert_action,
                    'source': f"Livello {level.get('id', level_idx)}, mossa {step_idx}"
                }
            else:
                # Se abbiamo già visto questo stato, controlliamo se l'azione è la stessa
                stored_action = state_map[state_key]['action']
                if stored_action != expert_action:
                    collisions_found += 1
                    print(f"\n💥 CONTRADDIZIONE TROVATA #{collisions_found}!")
                    print(f"  - Stato identico trovato in due punti diversi con azioni diverse:")
                    print(f"    1. Fonte: {state_map[state_key]['source']}")
                    print(f"       Azione richiesta: {action_idx_to_move[stored_action]}")
                    print(f"    2. Fonte: Livello {level.get('id', level_idx)}, mossa {step_idx}")
                    print(f"       Azione richiesta: {action_idx_to_move[expert_action]}")

            # Avanza l'ambiente per il prossimo step
            obs, _, _, _, _ = env.step(expert_action)

    if collisions_found == 0:
        print("\n✅ Nessuna collisione di stati trovata nel dataset.")
    else:
        print(f"\n🔥 Analisi completata. Trovate {collisions_found} collisioni totali.")

if __name__ == "__main__":
    LEVEL_FILES = ['json_levels/demo_LEVELS.json', 'json_levels/full_biy_LEVELS.json']
    all_levels = []
    for f in LEVEL_FILES:
        with open(f, 'r') as file:
            data = json.load(file)
            all_levels.extend(data.get('levels', data))
            
    find_state_collisions(all_levels, 20, 20)