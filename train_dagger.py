# train_dagger.py (Versione Ottimizzata con Fine-Tuning)

import json
import torch
import numpy as np
import os
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from baba_env import BabaEnv
from agents.behavioral_cloning_AGENT import BEHAVIORAL_CLONINGAgent

def load_all_levels(filepath: str) -> list:
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

def collect_dagger_data(agent: BEHAVIORAL_CLONINGAgent, levels: list, max_h: int, max_w: int):
    print("🔍 Esecuzione dell'agente per raccogliere i punti di errore...")
    agent.policy_net.eval()
    new_states, new_actions = [], []
    move_to_action_idx = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    for level in levels:
        env = BabaEnv([level['ascii']], max_height=max_h, max_width=max_w)
        obs, _ = env.reset()
        for move_char in level['solution'].lower():
            if move_char not in move_to_action_idx: continue
            expert_action = move_to_action_idx[move_char]
            state_tensor = torch.tensor(np.array(obs), device=agent.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted_action = agent.policy_net(state_tensor).argmax().item()
            if predicted_action != expert_action:
                new_states.append(obs)
                new_actions.append(expert_action)
            obs, _, terminated, truncated, _ = env.step(expert_action)
            if terminated or truncated: break
    print(f"💡 Raccolti {len(new_states)} nuovi punti di dati di correzione.")
    return np.array(new_states, dtype=np.float32), np.array(new_actions, dtype=np.int64)

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVELS_FILES = ['json_levels/demo_LEVELS.json', 'json_levels/full_biy_LEVELS.json']
    MODEL_PATH_TEMPLATE = 'dagger_baba_iter_{}.pth'
    FINAL_MODEL_NAME = 'dagger_baba_final.pth'

    DAGGER_ITERATIONS = 15
    EPOCHS_PER_ITERATION = 60  # Più epoche per dare tempo al modello più grande di imparare
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-5 # Un learning rate più basso per un fine-tuning più stabile

    MAX_H, MAX_W = 20, 20

    print("🚀 === Avvio Training con DAgger OTTIMIZZATO === 🚀\n")

    all_levels = []
    for file_path in LEVELS_FILES:
        all_levels.extend(load_all_levels(file_path))
    if not all_levels:
        print("❌ Nessun livello trovato. Interruzione.")
        exit()

    # Inizializza l'agente che useremo per il training
    training_agent = BEHAVIORAL_CLONINGAgent()
    dummy_env = BabaEnv([], MAX_H, MAX_W)
    training_agent.initialize_for_training(dummy_env.observation_space.shape[0], MAX_H, MAX_W, dummy_env.action_space.n, LEARNING_RATE)
    
    # Addestramento iniziale sul dataset base
    print("\n--- Fase 1: Addestramento Iniziale ---")
    X_train, y_train = generate_initial_data(all_levels, MAX_H, MAX_W)
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(40): # Un buon numero di epoche per il training iniziale
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            batch_states, batch_actions = batch_states.to(training_agent.device), batch_actions.to(training_agent.device)
            training_agent.optimizer.zero_grad()
            outputs = training_agent.policy_net(batch_states)
            loss = training_agent.criterion(outputs, batch_actions)
            loss.backward()
            training_agent.optimizer.step()
            total_loss += loss.item()
        print(f"  Epoca Iniziale {epoch+1}/40 | Loss media: {total_loss / len(dataloader):.6f}")
    
    # Salva il primo modello
    initial_model_path = MODEL_PATH_TEMPLATE.format(0)
    training_agent.save_model(initial_model_path)

    # Ciclo principale di DAgger
    print("\n--- Fase 2: Ciclo di Fine-Tuning con DAgger ---")
    for i in range(DAGGER_ITERATIONS):
        print(f"\n{'='*20} DAgger Iterazione {i+1}/{DAGGER_ITERATIONS} {'='*20}")
        
        # Carica il modello dell'iterazione precedente per la raccolta dati
        current_model_path = MODEL_PATH_TEMPLATE.format(i)
        collection_agent = BEHAVIORAL_CLONINGAgent(model_path=current_model_path)
        collection_agent.initialize_for_inference()

        X_new, y_new = collect_dagger_data(collection_agent, all_levels, MAX_H, MAX_W)

        if len(X_new) == 0:
            print("\n🎉🎉🎉 L'agente non ha commesso errori! Training completato. 🎉🎉🎉")
            os.rename(current_model_path, FINAL_MODEL_NAME)
            break

        print(f"➕ Aggregazione dataset: {len(X_train)} (esistenti) + {len(X_new)} (nuovi) = {len(X_train) + len(X_new)} campioni totali.")
        X_train = np.concatenate((X_train, X_new))
        y_train = np.concatenate((y_train, y_new))

        print(f"🧠 Fine-tuning del modello per {EPOCHS_PER_ITERATION} epoche...")
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Usa lo stesso agente e continua ad addestrarlo (fine-tuning)
        training_agent.policy_net.train() 
        scheduler = StepLR(training_agent.optimizer, step_size=20, gamma=0.7)

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
            scheduler.step()
            print(f"  Epoca {epoch+1}/{EPOCHS_PER_ITERATION} | Loss: {total_loss / len(dataloader):.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Salva il modello di questa iterazione
        next_model_path = MODEL_PATH_TEMPLATE.format(i+1)
        training_agent.save_model(next_model_path)
    else: # Se il ciclo for finisce senza 'break'
        print(f"\n⚠️ DAgger ha completato tutte le {DAGGER_ITERATIONS} iterazioni senza raggiungere 0 errori.")
        final_path = MODEL_PATH_TEMPLATE.format(DAGGER_ITERATIONS)
        os.rename(final_path, FINAL_MODEL_NAME)

    print(f"\n✨ Processo DAgger completato! Il modello finale è '{FINAL_MODEL_NAME}'. ✨")