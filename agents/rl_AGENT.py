# rl_agent.py (Versione Finale Unificata)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple
import numpy as np
from typing import List, Optional

from base_agent import BaseAgent
from baba_env import BabaEnv 
from baba import GameState, Direction

# Le classi 'Transition', 'ReplayMemory', e 'DQN' rimangono invariate
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_channels, height, width, n_actions):
        super(DQN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        def conv_size_out(size, kernel_size=3, padding=1): return (size + 2 * padding - kernel_size) // 1 + 1
        conv_w, conv_h = conv_size_out(width), conv_size_out(height)
        self.fc_stack = nn.Sequential(
            nn.Linear(conv_w * conv_h * 64, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        x = self.conv_stack(x)
        return self.fc_stack(x.view(x.size(0), -1))

# --- Agente RL Unificato ---
class RLAgent(BaseAgent):
    # ✅ __init__ ora accetta 'env' come argomento OPZIONALE.
    def __init__(self, env: Optional[BabaEnv] = None):
        super().__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il dispositivo: {self.device}")

        # Parametri di addestramento e inferenza
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.LR = 1e-4
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.MODEL_PATH = 'dqn_baba_general_model.pth'
        self.MODEL_MAX_H = 20
        self.MODEL_MAX_W = 20

        # Inizializza le variabili a None. Verranno create al momento del bisogno.
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = None
        self.steps_done = 0

        # ✅ Se 'env' è fornito (caso di addestramento), inizializza tutto subito.
        if self.env is not None:
            self._initialize_for_training()

    def _initialize_for_training(self):
        """Inizializza le componenti necessarie per l'addestramento."""
        n_actions = self.env.action_space.n
        self.policy_net = DQN(self.env.num_object_types, self.env.max_height, self.env.max_width, n_actions).to(self.device)
        self.target_net = DQN(self.env.num_object_types, self.env.max_height, self.env.max_width, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        print("Agente inizializzato per l'addestramento.")
    
    def _initialize_for_inference(self):
        """Inizializza le componenti necessarie per l'inferenza (gioco)."""
        # Creiamo un ambiente temporaneo solo per conoscere il numero di canali
        # Questo è un piccolo trucco per non dover hardcodare questo valore
        dummy_env = BabaEnv([], self.MODEL_MAX_H, self.MODEL_MAX_W)
        num_types = dummy_env.num_object_types
        
        self.policy_net = DQN(num_types, self.MODEL_MAX_H, self.MODEL_MAX_W, 4).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
        except FileNotFoundError:
             print(f"❌ ERRORE: File del modello non trovato in '{self.MODEL_PATH}'.")
             self.policy_net = None
             return
        self.policy_net.eval()
        print("Agente inizializzato per l'inferenza.")

    # --- Metodi per l'Addestramento ---

    def train(self, num_episodes=500):
        if self.env is None:
            raise ValueError("L'agente deve essere inizializzato con un ambiente per l'addestramento.")
        
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for t in range(200):
                action_tensor = self._select_action_train(state)
                action = action_tensor.item()
                observation, reward, terminated, _, _ = self.env.step(action)
                total_reward += reward
                reward_tensor = torch.tensor([reward], device=self.device)
                next_state = None if terminated else observation
                self.memory.push(state, action_tensor, next_state, reward_tensor)
                state = next_state
                self._optimize_model()
                self._update_target_net()
                if terminated: break
            print(f"Episodio {i_episode+1}/{num_episodes} | Passi: {t+1} | Ricompensa: {total_reward:.2f}")

    def _select_action_train(self, state):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        
        if any(non_final_mask):
             non_final_next_states = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        if any(non_final_mask):
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    # --- Metodo per Giocare (Inferenza) ---

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        if self.policy_net is None:
            self._initialize_for_inference()
        if self.policy_net is None:
            return None # Il caricamento del modello è fallito

        initial_state_ascii = "\n".join("".join(row) for row in initial_state.orig_map)
        env = BabaEnv([initial_state_ascii], self.MODEL_MAX_H, self.MODEL_MAX_W)
        state, _ = env.reset()
        
        solution_path = []
        for step in range(200):
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(0)
                action = self.policy_net(state_tensor).max(1)[1].item()
            observation, reward, terminated, _, _ = env.step(action)
            state = observation
            solution_path.append(env.action_map[action])
            if terminated:
                return solution_path if reward > 0 else []
        return []