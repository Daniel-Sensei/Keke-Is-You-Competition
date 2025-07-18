# behavioral_cloning_AGENT.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List
import os

from base_agent import BaseAgent
from baba_env import BabaEnv
from baba import GameState, Direction

class DQN(nn.Module):
    """
    La stessa architettura di rete neurale che hai già definito.
    È perfettamente adatta per un task di classificazione.
    """
    def __init__(self, n_channels, height, width, n_actions):
        super(DQN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        conv_w, conv_h = width, height
        
        self.fc_stack = nn.Sequential(
            nn.Linear(conv_w * conv_h * 128, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.fc_stack(x)

class BEHAVIORAL_CLONINGAgent(BaseAgent):
    def __init__(self, model_path='behavioral_cloning_baba.pth'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il dispositivo: {self.device}")

        self.MODEL_PATH = model_path
        self.MODEL_MAX_H = 20 # Assumiamo dimensioni massime, da adattare se necessario
        self.MODEL_MAX_W = 20

        self.policy_net = None
        self.optimizer = None
        self.criterion = None

    def initialize_for_training(self, n_channels, height, width, n_actions, learning_rate=1e-4):
        """Inizializza la rete, l'ottimizzatore e la funzione di loss per l'addestramento."""
        self.policy_net = DQN(n_channels, height, width, n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss() # Loss per la classificazione
        self.policy_net.train() # Metti la rete in modalità training
        print("Agente inizializzato per l'addestramento (Behavioral Cloning).")

    def initialize_for_inference(self):
        """Carica un modello addestrato per giocare."""
        # Creiamo un ambiente dummy per conoscere le dimensioni
        dummy_env = BabaEnv([], self.MODEL_MAX_H, self.MODEL_MAX_W)
        total_channels = dummy_env.observation_space.shape[0]
        n_actions = dummy_env.action_space.n

        self.policy_net = DQN(total_channels, self.MODEL_MAX_H, self.MODEL_MAX_W, n_actions).to(self.device)
        
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.policy_net.eval() # Metti la rete in modalità valutazione
            print(f"Modello caricato con successo da '{self.MODEL_PATH}'.")
        except FileNotFoundError:
            print(f"❌ ERRORE: File del modello non trovato in '{self.MODEL_PATH}'.")
            self.policy_net = None
            
    def save_model(self):
        """Salva i pesi del modello addestrato."""
        if self.policy_net:
            torch.save(self.policy_net.state_dict(), self.MODEL_PATH)
            print(f"💾 Modello salvato con successo in '{self.MODEL_PATH}'.")

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        """
        Metodo di inferenza per giocare.
        Prende uno stato, interroga la rete e restituisce la mossa migliore predetta.
        """
        if self.policy_net is None:
            self.initialize_for_inference()
        if self.policy_net is None:
            return None
        
        # Crea un ambiente temporaneo per questo specifico livello
        initial_state_ascii = "\n".join("".join(row) for row in initial_state.orig_map)
        env = BabaEnv([initial_state_ascii], self.MODEL_MAX_H, self.MODEL_MAX_W)
        state, _ = env.reset()
        
        solution_path = []
        for step in range(min(iterations, 200)): # Limite di passi per evitare loop infiniti
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=self.device, 
                                          dtype=torch.float32).unsqueeze(0)
                # Ottieni l'azione con la probabilità più alta (la previsione della rete)
                action_probs = self.policy_net(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()

            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            solution_path.append(env.action_map[action])

            if info.get('won', False):
                return solution_path # Soluzione trovata!
            if terminated or truncated:
                break
                
        return [] # Soluzione non trovata