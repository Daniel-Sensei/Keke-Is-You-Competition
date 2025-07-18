# train_dagger.py

import json
import torch
import numpy as np
import os
import time
from torch.utils.data import TensorDataset, DataLoader

# Assicurati che questi file siano nella stessa directory
from baba_env import BabaEnv
from behavioral_cloning_AGENT import BehavioralCloningAgent

def load_all_levels(filepath: str) -> list:
    """Carica tutti i livelli da un file JSON che hanno una soluzione."""
    all_levels = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            levels = data.get('levels', data)
            for level in levels:
                if 'ascii' in level and 'solution' in level and level['solution']:
                    all_levels.append(level)
            print(f"✅ Caricati {len(all_levels)} livelli con soluzione da {filepath}")
    except Exception as e:
        print(f"❌ Errore nel caricare {filepath}: {e}")
    return all_levels

def generate_initial_data(levels: list, max_h: int, max_w: int):
    """Genera il dataset iniziale basato solo sulle soluzioni perfette."""
    print("\n🔄 Generazione del dataset di training iniziale...")
    states, actions = [], []
    move_to_action_idx = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    
    for level in levels:
        env = BabaEnv([level['ascii']], max_height=max_h, max_width=max_w)
        obs, _ = env.reset()
        for move_char in level['solution'].lower():
            if move_char not in move_to_action_idx: continue
            action_idx = move_to_action_idx[move_char]
            states.append(obs)
            actions.append(action_idx)
            obs, _, terminated, truncated, _ = env.step(action_idx)
            if terminated or truncated: break
            
    print(f"✅ Dataset iniziale creato con {len(states)} coppie (stato, azione).")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

def collect_dagger_data(agent: BehavioralCloningAgent, levels: list, max_h: int, max_w: int):
    """
    Esegue l'agente corrente, identifica i suoi errori e raccoglie i dati per correggerli.
    """
    print("🔍 Esecuzione dell'agente per raccogliere i punti di errore...")
    agent.policy_net.eval() # Assicurati che l'agente sia in modalità valutazione
    
    new_states, new_actions = [], []
    move_to_action_idx = {'u': 0, 'd': 1, 'l': 2, 'r': 3}

    for level in levels:
        env = BabaEnv([level['ascii']], max_height=max_h, max_width=max_w)
        obs, _ = env.reset()
        
        for move_char in level['solution'].lower():
            if move_char not in move_to_action_idx: continue
            
            # L'azione corretta secondo l'esperto
            expert_action = move_to_action_idx[move_char]

            # L'azione che l'agente avrebbe scelto
            state_tensor = torch.tensor(np.array(obs), device=agent.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_action = agent.policy_net(state_tensor).argmax().item()

            # Se l'agente sbaglia, salviamo la correzione
            if predicted_action != expert_action:
                new_states.append(obs)
                new_actions.append(expert_action)
            
            # IMPORTANTE: avanziamo sempre con la mossa dell'ESPERTO per rimanere sulla traiettoria ottimale
            obs, _, terminated, truncated, _ = env.step(expert_action)
            if terminated or truncated: break
            
    print(f"💡 Raccolti {len(new_states)} nuovi punti di dati di correzione.")
    return np.array(new_states, dtype=np.float32), np.array(new_actions, dtype=np.int64)

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVELS_FILES = [
        'json_levels/demo_LEVELS.json',
        'json_levels/full_biy_LEVELS.json'
    ]
    INITIAL_MODEL_PATH = 'behavioral_cloning_baba.pth' # Il modello che hai già addestrato
    FINAL_MODEL_PATH = 'dagger_baba_final.pth'

    # Parametri di DAgger e training
    DAGGER_ITERATIONS = 10
    EPOCHS_PER_ITERATION = 50 # Meno epoche per iterazione, dato che il dataset cresce
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4

    MAX_H, MAX_W = 20, 20

    print("🚀 === Avvio Training con DAgger (Dataset Aggregation) === 🚀\n")

    # 1. Carica tutti i livelli
    all_levels = []
    for file_path in LEVELS_FILES:
        all_levels.extend(load_all_levels(file_path))
    if not all_levels:
        print("❌ Nessun livello trovato. Interruzione.")
        exit()

    # 2. Genera il dataset di training iniziale dalle soluzioni
    X_train, y_train = generate_initial_data(all_levels, MAX_H, MAX_W)

    # 3. Inizializza l'agente e carica il modello pre-addestrato
    current_agent = BehavioralCloningAgent(model_path=INITIAL_MODEL_PATH)
    current_agent.initialize_for_inference() # Carica i pesi
    if current_agent.policy_net is None:
        print(f"Errore: impossibile caricare il modello iniziale da '{INITIAL_MODEL_PATH}'. Assicurati che esista.")
        exit()

    # 4. Ciclo principale di DAgger
    for i in range(DAGGER_ITERATIONS):
        print(f"\n{'='*20} DAgger Iterazione {i+1}/{DAGGER_ITERATIONS} {'='*20}")

        # Fase di raccolta: trova gli errori dell'agente corrente
        X_new, y_new = collect_dagger_data(current_agent, all_levels, MAX_H, MAX_W)

        # Se non ci sono nuovi dati, l'agente è perfetto. Abbiamo finito.
        if len(X_new) == 0:
            print("\n🎉🎉🎉 L'agente non ha commesso errori! Training completato con successo. 🎉🎉🎉")
            break

        # Fase di aggregazione: aggiungi i nuovi dati a quelli esistenti
        print(f"➕ Aggregazione dataset: {len(X_train)} (esistenti) + {len(X_new)} (nuovi) = {len(X_train) + len(X_new)} campioni totali.")
        X_train = np.concatenate((X_train, X_new))
        y_train = np.concatenate((y_train, y_new))

        # Fase di ri-addestramento: addestra un nuovo modello sul dataset aggregato
        print(f"🧠 Ri-addestramento del modello per {EPOCHS_PER_ITERATION} epoche...")
        
        # Prepara il DataLoader con i dati aggiornati
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Inizializza un nuovo agente per il training (o re-inizializza)
        # Inizializzare da zero è spesso più stabile
        training_agent = BehavioralCloningAgent()
        dummy_env = BabaEnv([], MAX_H, MAX_W)
        training_agent.initialize_for_training(dummy_env.observation_space.shape[0], MAX_H, MAX_W, dummy_env.action_space.n, LEARNING_RATE)

        start_time = time.time()
        for epoch in range(EPOCHS_PER_ITERATION):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                batch_states, batch_actions = batch_states.to(training_agent.device), batch_actions.to(training_agent.device)
                training_agent.optimizer.zero_grad()
                outputs = training_agent.policy_net(batch_states)
                loss = training_agent.criterion(outputs, batch_actions)
                loss.backward()
                training_agent.optimizer.step()
                total_loss += loss.item()
            print(f"  Epoca {epoch+1}/{EPOCHS_PER_ITERATION} | Loss media: {total_loss / len(dataloader):.6f}")
        
        # Il modello appena addestrato diventa il nostro nuovo "agente corrente" per la prossima iterazione
        current_agent = training_agent
        print(f"  Tempo di training per questa iterazione: {time.time() - start_time:.2f}s")
    
    # 5. Salva il modello finale
    print(f"\n💾 Salvataggio del modello finale e più robusto in '{FINAL_MODEL_PATH}'...")
    current_agent.save_model()
    os.rename(current_agent.MODEL_PATH, FINAL_MODEL_PATH) # Rinomina per chiarezza
    
    print(f"\n✨ Processo DAgger completato! Il modello finale '{FINAL_MODEL_PATH}' è pronto. ✨")