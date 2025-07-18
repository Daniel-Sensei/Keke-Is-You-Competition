# train_behavioral_cloning.py (Versione FINALE per ResNet con dati non ambigui)

import json
import torch
import numpy as np
import os
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from baba_env import BabaEnv
from agents.behavioral_cloning_AGENT import BEHAVIORAL_CLONINGAgent

def load_all_levels(filepaths: list) -> list:
    all_levels = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                levels = data.get('levels', data)
                valid_levels = [lvl for lvl in levels if 'ascii' in lvl and 'solution' in lvl and lvl['solution']]
                all_levels.extend(valid_levels)
                print(f"✅ Caricati {len(valid_levels)} livelli validi da {filepath}")
        except Exception as e:
            print(f"❌ Errore nel caricare {filepath}: {e}")
    print(f"✅ Totale livelli caricati con soluzione: {len(all_levels)}")
    return all_levels

def generate_training_data(levels: list, max_h: int, max_w: int):
    states, actions = [], []
    move_to_action_idx = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    print("\n🔄 Generazione dati di training (con step-count)...")
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
    print(f"✅ Generazione completata. Totale coppie (stato, azione): {len(states)}\n")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVEL_FILES = ['json_levels/demo_LEVELS.json', 'json_levels/full_biy_LEVELS.json']
    MODEL_SAVE_PATH = 'baba_final_agent.pth'
    
    # Parametri di addestramento intenso
    NUM_EPOCHS = 200 # Più tempo per convergere perfettamente
    BATCH_SIZE = 256
    MAX_LEARNING_RATE = 1e-3 

    MAX_H, MAX_W = 20, 20

    print("🚀 === Avvio Training FINALE con ResNet e Dati Disambiguati === 🚀\n")

    levels_with_solutions = load_all_levels(LEVEL_FILES)
    if not levels_with_solutions:
        print("❌ Nessun livello con soluzioni trovato. Interruzione.")
        exit()

    X_train, y_train = generate_training_data(levels_with_solutions, MAX_H, MAX_W)
    
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    agent = BEHAVIORAL_CLONINGAgent(model_path=MODEL_SAVE_PATH)
    
    dummy_env = BabaEnv([], MAX_H, MAX_W)
    agent.initialize_for_training(dummy_env.observation_space.shape[0], MAX_H, MAX_W, dummy_env.action_space.n, MAX_LEARNING_RATE)

    scheduler = OneCycleLR(agent.optimizer, max_lr=MAX_LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(dataloader))

    print(f"🎮 Inizio training per {NUM_EPOCHS} epoche...\n")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            batch_states, batch_actions = batch_states.to(agent.device), batch_actions.to(agent.device)
            agent.optimizer.zero_grad()
            outputs = agent.policy_net(batch_states)
            loss = agent.criterion(outputs, batch_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
            agent.optimizer.step()
            scheduler.step() 
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoca {epoch+1}/{NUM_EPOCHS} | Loss media: {avg_loss:.6f} | Learning Rate: {current_lr:.2e}")

    agent.save_model()
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Tempo totale di training: {elapsed_time:.2f} secondi")
    print(f"✨ Addestramento completato! Il modello '{MODEL_SAVE_PATH}' è pronto per il 100%. ✨")