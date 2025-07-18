# train_rl.py (Versione con Curriculum Learning)

import json
import torch
import numpy as np
from baba_env import BabaEnv
from agents.rl_AGENT import RLAgent

def load_and_categorize_levels(filepaths: list) -> dict:
    """Carica e classifica i livelli per difficoltà basata sulla lunghezza della soluzione."""
    categorized_levels = {1: [], 2: [], 3: []}
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for level in data['levels']:
                    solution_length = len(level.get('solution', ''))
                    if solution_length <= 15:
                        categorized_levels[1].append(level['ascii'])
                    elif 15 < solution_length <= 40:
                        categorized_levels[2].append(level['ascii'])
                    else: # > 40
                        categorized_levels[3].append(level['ascii'])
            print(f"Caricati e classificati {len(data['levels'])} livelli da {filepath}")
        except Exception as e:
            print(f"Errore nel caricare {filepath}: {e}")
    return categorized_levels

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVEL_FILES = ['json_levels/fdemo_LEVELS.json', 'json_levels/full_biy_LEVELS.json']
    NUM_EPISODES_TOTAL = 10000
    MODEL_SAVE_PATH = 'dqn_baba_curriculum_model.pth'
    
    print("--- Avvio Addestramento con Curriculum Learning ---")
    
    # 1. Carica e classifica tutti i livelli
    levels_by_difficulty = load_and_categorize_levels(LEVEL_FILES)
    all_maps = levels_by_difficulty[1] + levels_by_difficulty[2] + levels_by_difficulty[3]
    if not all_maps:
        print("Nessun livello caricato. Interruzione.")
        exit()
        
    max_h = max(len(m.split('\n')) for m in all_maps)
    max_w = max(len(line) for m in all_maps for line in m.split('\n'))
    print(f"\nDimensioni massime rilevate (padding): {max_h}x{max_w}")

    # 2. Inizializza l'ambiente e l'agente
    # Iniziamo con i livelli facili
    current_training_pool = levels_by_difficulty[1]
    env = BabaEnv(list_of_ascii_maps=current_training_pool, max_height=max_h, max_width=max_w)
    agent = RLAgent(env)
    
    # 3. Ciclo di addestramento a fasi
    episodes_per_phase = NUM_EPISODES_TOTAL // 3
    recent_rewards = []

    for phase in [1, 2, 3]:
        print(f"\n--- Inizio Fase di Addestramento {phase} ---")
        print(f"Numero di livelli nel pool di training: {len(current_training_pool)}")
        
        agent.train(num_episodes=episodes_per_phase)
        
        # Dopo ogni fase, espandi il pool di livelli se non è l'ultima fase
        if phase < 3:
            print(f"--- Fine Fase {phase}. Espansione del pool di livelli. ---")
            current_training_pool.extend(levels_by_difficulty[phase + 1])
            # Aggiorna l'ambiente dell'agente con il nuovo pool
            agent.env.list_of_ascii_maps = current_training_pool

    print("\nAddestramento completato.")

    # 4. Salva il modello finale
    print(f"Salvataggio del modello in '{MODEL_SAVE_PATH}'...")
    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print("Modello salvato con successo!")