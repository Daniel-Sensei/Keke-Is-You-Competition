# rl_AGENT.py (Versione Migliorata con Double DQN e Rete Più Profonda)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple
import numpy as np
from typing import List, Optional
import os

from base_agent import BaseAgent
from baba_env import BabaEnv 
from baba import GameState, Direction

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args): 
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size): 
        return random.sample(self.memory, batch_size)
    
    def __len__(self): 
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_channels, height, width, n_actions):
        super(DQN, self).__init__()
        
        # Rete convoluzionale più profonda
        self.conv_stack = nn.Sequential(
            # Layer 1
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3 (nuovo)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 4 (nuovo)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Con padding=1, le dimensioni rimangono invariate
        conv_w, conv_h = width, height
        
        # Rete fully connected più profonda
        self.fc_stack = nn.Sequential(
            nn.Linear(conv_w * conv_h * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout per regolarizzazione
            
            nn.Linear(512, 256),  # Layer aggiuntivo
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, n_actions)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.fc_stack(x)

class RLAgent(BaseAgent):
    def __init__(self, env: Optional[BabaEnv] = None):
        super().__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando il dispositivo: {self.device}")

        # Parametri di addestramento migliorati
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.LR = 1e-4
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 5000  # Decay più lento
        self.MEMORY_SIZE = 100000  # Buffer più grande
        self.UPDATE_TARGET_EVERY = 10  # Aggiorna target network ogni N step
        
        # Percorsi per salvare/caricare
        self.MODEL_PATH = 'dqn_baba_improved_model.pth'
        self.CHECKPOINT_PATH = 'checkpoints/'
        self.MODEL_MAX_H = 20
        self.MODEL_MAX_W = 20

        # Inizializza le variabili
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = None
        self.steps_done = 0
        self.update_counter = 0
        
        # Learning rate scheduler
        self.scheduler = None

        # Se env è fornito, inizializza per training
        if self.env is not None:
            self._initialize_for_training()

    def _initialize_for_training(self):
        """Inizializza le componenti necessarie per l'addestramento."""
        n_actions = self.env.action_space.n
        total_channels = self.env.observation_space.shape[0]
        
        self.policy_net = DQN(total_channels, self.env.max_height, 
                            self.env.max_width, n_actions).to(self.device)
        self.target_net = DQN(total_channels, self.env.max_height, 
                            self.env.max_width, n_actions).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        self.memory = ReplayMemory(self.MEMORY_SIZE)
        
        # Crea directory per checkpoint se non esiste
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        
        print("Agente inizializzato per l'addestramento con rete migliorata.")
    
    def _initialize_for_inference(self):
        """Inizializza le componenti necessarie per l'inferenza."""
        # Creiamo un ambiente dummy per conoscere le dimensioni
        dummy_env = BabaEnv([], self.MODEL_MAX_H, self.MODEL_MAX_W)
        total_channels = dummy_env.observation_space.shape[0]
        
        self.policy_net = DQN(total_channels, self.MODEL_MAX_H, 
                            self.MODEL_MAX_W, 4).to(self.device)
        try:
            checkpoint = torch.load(self.MODEL_PATH, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"Modello caricato dal checkpoint all'episodio {checkpoint.get('episode', 'N/A')}")
            else:
                self.policy_net.load_state_dict(checkpoint)
        except FileNotFoundError:
            print(f"❌ ERRORE: File del modello non trovato in '{self.MODEL_PATH}'.")
            self.policy_net = None
            return
        
        self.policy_net.eval()
        print("Agente inizializzato per l'inferenza.")

    def train(self, num_episodes=500, save_checkpoint_every=5000):
        if self.env is None:
            raise ValueError("L'agente deve essere inizializzato con un ambiente per l'addestramento.")
        
        best_avg_reward = float('-inf')
        episode_rewards = deque(maxlen=100)
        
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_loss = 0
            loss_count = 0
            
            for t in range(self.env.max_episode_steps):
                action_tensor = self._select_action_train(state)
                action = action_tensor.item()
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                
                reward_tensor = torch.tensor([reward], device=self.device)
                next_state = None if (terminated or truncated) else observation
                
                self.memory.push(state, action_tensor, next_state, reward_tensor)
                state = next_state
                
                # Ottimizza il modello
                loss = self._optimize_model()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                # Aggiorna target network periodicamente
                if self.update_counter % self.UPDATE_TARGET_EVERY == 0:
                    self._update_target_net()
                
                self.update_counter += 1
                
                if terminated or truncated:
                    break
            
            # Aggiorna learning rate
            self.scheduler.step()
            
            # Tracking
            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards)
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0
            
            # Stampa progressi
            if (i_episode + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Episodio {i_episode+1}/{num_episodes} | "
                      f"Passi: {t+1} | "
                      f"Reward: {total_reward:.2f} | "
                      f"Avg Reward (100ep): {avg_reward:.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {self._get_epsilon():.3f} | "
                      f"LR: {current_lr:.6f}")
            
            # Salva checkpoint
            if (i_episode + 1) % save_checkpoint_every == 0:
                self._save_checkpoint(i_episode + 1, avg_reward)
            
            # Salva se è il miglior modello
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                self._save_checkpoint(i_episode + 1, avg_reward, is_best=True)

    def _get_epsilon(self):
        """Calcola il valore corrente di epsilon per epsilon-greedy."""
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
               math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def _select_action_train(self, state):
        eps_threshold = self._get_epsilon()
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=self.device, 
                                          dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], 
                              device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None
            
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=self.device, dtype=torch.bool)
        
        if any(non_final_mask):
            non_final_next_states = torch.cat([torch.from_numpy(s).unsqueeze(0) 
                                             for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Q(s, a) corrente
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: usa policy_net per scegliere azioni, target_net per valutarle
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        if any(non_final_mask):
            with torch.no_grad():
                # Seleziona le azioni usando la policy network
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                # Valuta usando la target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze()
        
        # Target Q-value
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        # Calcola loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Ottimizza
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def _update_target_net(self):
        """Soft update della target network."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + \
                                       target_net_state_dict[key] * (1 - self.TAU)
        
        self.target_net.load_state_dict(target_net_state_dict)

    def _save_checkpoint(self, episode, avg_reward, is_best=False):
        """Salva un checkpoint del modello."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'avg_reward': avg_reward,
            'steps_done': self.steps_done,
            'epsilon': self._get_epsilon()
        }
        
        if is_best:
            path = self.MODEL_PATH
            print(f"💾 Salvato miglior modello con avg_reward={avg_reward:.2f}")
        else:
            path = os.path.join(self.CHECKPOINT_PATH, f'checkpoint_ep{episode}.pth')
            print(f"💾 Salvato checkpoint all'episodio {episode}")
        
        torch.save(checkpoint, path)

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        """Metodo per giocare (inferenza)."""
        if self.policy_net is None:
            self._initialize_for_inference()
        if self.policy_net is None:
            return None
        
        # Crea ambiente temporaneo per questo livello
        initial_state_ascii = "\n".join("".join(row) for row in initial_state.orig_map)
        env = BabaEnv([initial_state_ascii], self.MODEL_MAX_H, self.MODEL_MAX_W)
        state, _ = env.reset()
        
        solution_path = []
        for step in range(min(iterations, 200)):
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=self.device, 
                                          dtype=torch.float32).unsqueeze(0)
                action = self.policy_net(state_tensor).max(1)[1].item()
            
            observation, reward, terminated, truncated, _ = env.step(action)
            state = observation
            solution_path.append(env.action_map[action])
            
            if terminated:
                return solution_path if reward > 0 else []
            if truncated:
                break
                
        return []