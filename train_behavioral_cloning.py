# train_behavioral_cloning.py

import json
import torch
import numpy as np
import os
import time
from torch.utils.data import TensorDataset, DataLoader

from baba_env import BabaEnv
from baba import Direction
from agents.behavioral_cloning_AGENT import BEHAVIORAL_CLONINGAgent

def load_all_levels(filepaths: list) -> list:
    """Carica tutti i livelli da una lista di file JSON."""
    all_levels = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Gestisce sia il formato dizionario {'levels': [...]} che lista [...]
                levels = data.get('levels', data)
                for level in levels:
                    if 'ascii' in level and 'solution' in level and level['solution']:
                        all_levels.append(level)
                print(f"✅ Caricati {len(levels)} livelli da {filepath}")
        except Exception as e:
            print(f"❌ Errore nel caricare {filepath}: {e}")
    print(f"✅ Totale livelli caricati con soluzione: {len(all_levels)}")
    return all_levels

def generate_training_data(levels: list, max_h: int, max_w: int):
    """
    Genera coppie (stato, azione) imitando le soluzioni fornite.
    """
    states = []
    actions = []

    # Mappa per convertire il carattere della mossa in un indice intero
    # NOTA: Questa mappa è l'inverso di quella in baba_env.py
    # action_map = {0: Direction.Up, 1: Direction.Down, 2: Direction.Left, 3: Direction.Right}
    move_to_action_idx = {
        'u': 0, 'U': 0,
        'd': 1, 'D': 1,
        'l': 2, 'L': 2,
        'r': 3, 'R': 3
        # 's' (wait) viene ignorato perché non è un'azione del nostro agente
    }
    
    print("\n🔄 Generazione dati di training dalle soluzioni...")
    total_pairs = 0
    for i, level in enumerate(levels):
        ascii_map = level['ascii']
        solution = level['solution']
        
        # Crea un ambiente solo per questo livello
        env = BabaEnv([ascii_map], max_height=max_h, max_width=max_w)
        obs, _ = env.reset()

        for move_char in solution:
            if move_char.lower() not in move_to_action_idx:
                continue

            action_idx = move_to_action_idx[move_char.lower()]
            
            # Aggiungi la coppia (stato corrente, azione corretta)
            states.append(obs)
            actions.append(action_idx)
            total_pairs += 1

            # Esegui l'azione per ottenere lo stato successivo
            obs, _, terminated, truncated, _ = env.step(action_idx)

            if terminated or truncated:
                break # Passa al livello successivo se finisce prima
        
        if (i + 1) % 50 == 0:
            print(f"  ... processati {i+1}/{len(levels)} livelli.")

    print(f"✅ Generazione completata. Totale coppie (stato, azione): {total_pairs}\n")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVEL_FILES = ['json_levels/demo_LEVELS.json', 'json_levels/full_biy_LEVELS.json']
    MODEL_SAVE_PATH = 'behavioral_cloning_baba.pth'
    
    # Parametri di addestramento
    NUM_EPOCHS = 20  # Aumenta se la loss non converge
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4

    # Dimensioni massime della mappa (ricalcolate per sicurezza)
    MAX_H = 20
    MAX_W = 20

    print("🚀 === Avvio Training con Behavioral Cloning per Baba is You === 🚀\n")

    # 1. Carica i livelli
    levels_with_solutions = load_all_levels(LEVEL_FILES)
    if not levels_with_solutions:
        print("❌ Nessun livello con soluzioni trovato. Interruzione.")
        exit()

    # 2. Genera i dati di training
    X_train, y_train = generate_training_data(levels_with_solutions, MAX_H, MAX_W)
    
    # 3. Prepara il DataLoader di PyTorch per un training efficiente
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Inizializza l'agente per l'addestramento
    agent = BEHAVIORAL_CLONINGAgent(model_path=MODEL_SAVE_PATH)
    
    # Ottieni le dimensioni dall'ambiente per inizializzare la rete
    dummy_env = BabaEnv([], MAX_H, MAX_W)
    n_channels = dummy_env.observation_space.shape[0]
    n_actions = dummy_env.action_space.n
    agent.initialize_for_training(n_channels, MAX_H, MAX_W, n_actions, LEARNING_RATE)

    # 5. Ciclo di addestramento
    print(f"🎮 Inizio training per {NUM_EPOCHS} epoche...\n")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            # Sposta i dati sul dispositivo corretto (CPU o GPU)
            batch_states = batch_states.to(agent.device)
            batch_actions = batch_actions.to(agent.device)

            # Azzera i gradienti
            agent.optimizer.zero_grad()

            # Forward pass: ottieni le previsioni della rete
            outputs = agent.policy_net(batch_states)
            
            # Calcola la loss
            loss = agent.criterion(outputs, batch_actions)
            
            # Backward pass e ottimizzazione
            loss.backward()
            agent.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoca {epoch+1}/{NUM_EPOCHS} | Loss media: {avg_loss:.6f}")

    # 6. Salva il modello finale
    agent.save_model()
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Tempo totale di training: {elapsed_time:.2f} secondi")
    print(f"✨ Addestramento completato! Il modello '{MODEL_SAVE_PATH}' è pronto. ✨")