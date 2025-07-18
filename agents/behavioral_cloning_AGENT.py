# behavioral_cloning_AGENT.py (Versione con Architettura ResNet e Beam Search Corretta)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import os
import copy

from base_agent import BaseAgent
from baba_env import BabaEnv, check_win, advance_game_state
from baba import GameState, Direction

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + identity)
        return out

class DQN_ResNet(nn.Module):
    def __init__(self, n_channels, height, width, n_actions):
        super(DQN_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_stack = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, n_actions))
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides, layers = [stride] + [1] * (num_blocks - 1), []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.fc_stack(out)

class BEHAVIORAL_CLONINGAgent(BaseAgent):
    def __init__(self, model_path='baba_final_agent.pth'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il dispositivo: {self.device}")
        self.MODEL_PATH, self.MODEL_MAX_H, self.MODEL_MAX_W = model_path, 20, 20
        self.policy_net, self.optimizer, self.criterion = None, None, None

    def initialize_for_training(self, n_channels, height, width, n_actions, learning_rate=1e-4):
        self.policy_net = DQN_ResNet(n_channels, height, width, n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.policy_net.train()
        print("Agente inizializzato per l'addestramento con la rete DQN_ResNet.")

    def initialize_for_inference(self):
        dummy_env = BabaEnv([], self.MODEL_MAX_H, self.MODEL_MAX_W)
        total_channels, n_actions = dummy_env.observation_space.shape[0], dummy_env.action_space.n
        self.policy_net = DQN_ResNet(total_channels, self.MODEL_MAX_H, self.MODEL_MAX_W, n_actions).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.policy_net.eval()
            print(f"Modello caricato con successo da '{self.MODEL_PATH}'.")
        except FileNotFoundError:
            print(f"❌ ERRORE: File del modello non trovato in '{self.MODEL_PATH}'.")
            self.policy_net = None
    
    def save_model(self, path=None):
        save_path = path if path else self.MODEL_PATH
        if self.policy_net:
            torch.save(self.policy_net.state_dict(), save_path)
            print(f"💾 Modello salvato con successo in '{save_path}'.")

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        """
        Esegue una ricerca della soluzione usando l'algoritmo Beam Search per migliorare
        la robustezza rispetto a una ricerca greedy.
        """
        if self.policy_net is None:
            self.initialize_for_inference()
        if self.policy_net is None:
            return None

        # --- Inizializzazione ---
        base_env = BabaEnv(["\n".join("".join(row) for row in initial_state.orig_map)], self.MODEL_MAX_H, self.MODEL_MAX_W)
        base_env.reset()
        
        # --- Parametri della Beam Search ---
        # Correzione: assicurati che beam_width non sia maggiore del numero di azioni disponibili
        n_actions = base_env.action_space.n
        beam_width = min(5, n_actions) # <-- RIGA MODIFICATA E SPOSTATA
        max_steps = min(iterations, 100)


        # Il "beam" contiene tuple di: (log_probabilità_cumulativa, percorso_azioni, stato_ambiente_clonato)
        beam = [(0.0, [], initial_state.copy())]

        for step in range(max_steps):
            if not beam:
                break

            potential_candidates = []
            for log_prob, path, game_state in beam:
                if check_win(game_state):
                    potential_candidates.append((log_prob, path, game_state))
                    continue
                
                temp_env = BabaEnv([base_env.current_ascii_map], self.MODEL_MAX_H, self.MODEL_MAX_W)
                temp_env.state = game_state
                temp_env.episode_steps = len(path)
                obs = temp_env._get_obs()

                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32).unsqueeze(0)
                    action_log_probs = F.log_softmax(self.policy_net(state_tensor), dim=1)
                    
                    # Ora questa chiamata è sicura perché beam_width <= n_actions
                    top_log_probs, top_actions = torch.topk(action_log_probs, beam_width)

                    for i in range(beam_width):
                        action_idx = top_actions[0, i].item()
                        current_log_prob = top_log_probs[0, i].item()
                        
                        next_state = game_state.copy()
                        direction = base_env.action_map[action_idx]
                        next_state = advance_game_state(direction, next_state)

                        new_path = path + [direction]
                        new_log_prob = log_prob + current_log_prob
                        potential_candidates.append((new_log_prob, new_path, next_state))

            def score_path(candidate):
                log_prob, path, state = candidate
                if check_win(state): return float('inf')
                return log_prob / len(path) if path else -float('inf')

            sorted_candidates = sorted(potential_candidates, key=score_path, reverse=True)
            beam = [(log_prob, path, state) for log_prob, path, state in sorted_candidates[:beam_width]]

            if beam and check_win(beam[0][2]):
                print(f"✨ Soluzione trovata al passo {step + 1} con Beam Search!")
                return beam[0][1]

        print("❌ Nessuna soluzione trovata entro il limite di passi con Beam Search.")
        return []