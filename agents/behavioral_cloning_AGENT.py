# behavioral_cloning_AGENT.py (Versione con Correzione TypeError)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

from base_agent import BaseAgent
from baba_env import BabaEnv, check_win, advance_game_state
from baba import GameState, Direction, GameObj, GameObjectType

# --- Architettura della Rete (ResNet) ---
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


# --- Agente Principale ---
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
    
    # --- METODI EURISTICI (dall'agente A*) ---
    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _find_word_objects(self, state: GameState, word_name: str) -> List[GameObj]:
        return [w for w in state.words if w.name == word_name]

    def _analyze_current_rules(self, state: GameState) -> Dict[str, List[str]]:
        rule_analysis = {'you_rules': [], 'win_rules': [], 'other_rules': []}
        if not hasattr(state, 'rules'): return rule_analysis
        for rule in state.rules:
            if '-is-you' in rule: rule_analysis['you_rules'].append(rule)
            elif '-is-win' in rule: rule_analysis['win_rules'].append(rule)
            else: rule_analysis['other_rules'].append(rule)
        return rule_analysis

    def _heuristic(self, state: GameState) -> float:
        if check_win(state): return 0
        if not state.players: return float('inf')

        rule_analysis = self._analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        has_win = bool(rule_analysis['win_rules'])

        if has_you and has_win:
            if state.winnables:
                min_dist = min(self._manhattan_distance((p.x, p.y), (w.x, w.y)) for p in state.players for w in state.winnables)
                return min_dist
            else: return 100.0
        
        if not has_you:
            cost = self._estimate_rule_formation_cost(state, 'you')
            return 500.0 + cost

        if not has_win:
            cost = self._estimate_rule_formation_cost(state, 'win')
            return 100.0 + cost

        return float('inf')

    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = self._find_word_objects(state, 'is')
        property_words = self._find_word_objects(state, target_property_name)

        if not nouns or not is_words or not property_words: return float('inf')

        min_formation_dist = float('inf')
        for noun in nouns:
            if noun.name == target_property_name: continue
            dist_to_is = min(self._manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words)
            dist_to_prop = min(self._manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words)
            min_formation_dist = min(min_formation_dist, dist_to_is + dist_to_prop)
        
        return min_formation_dist if min_formation_dist != float('inf') else float('inf')

    # --- METODO DI RICERCA AGGIORNATO ---
    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        if self.policy_net is None: self.initialize_for_inference()
        if self.policy_net is None: return None

        base_env = BabaEnv(["\n".join("".join(row) for row in initial_state.orig_map)], self.MODEL_MAX_H, self.MODEL_MAX_W)
        base_env.reset()
        
        n_actions = base_env.action_space.n
        beam_width = 20
        max_path_length = 200
        heuristic_weight = 0.05

        beam = [(0.0, 0.0, [], initial_state.copy())]
        visited_states = {}

        for step in range(max_path_length):
            if not beam: break

            potential_candidates = []
            for score, model_log_prob, path, game_state in beam:
                if check_win(game_state):
                    print(f"✨ Soluzione trovata al passo {len(path)}!")
                    return path

                state_representation = tuple(sorted([str((obj.name, obj.x, obj.y)) for obj in game_state.phys + game_state.words] + game_state.rules))
                if state_representation in visited_states and visited_states[state_representation] >= model_log_prob:
                    continue
                visited_states[state_representation] = model_log_prob
                
                temp_env = BabaEnv([base_env.current_ascii_map], self.MODEL_MAX_H, self.MODEL_MAX_W)
                temp_env.state = game_state
                temp_env.episode_steps = len(path)
                obs = temp_env._get_obs()

                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32).unsqueeze(0)
                    action_log_probs = F.log_softmax(self.policy_net(state_tensor), dim=1)
                    
                    for action_idx in range(n_actions):
                        # <-- CORREZIONE APPLICATA QUI
                        # Chiama la funzione con argomenti posizionali, come richiesto dalla sua definizione
                        next_state = advance_game_state(base_env.action_map[action_idx], game_state.copy())
                        
                        h_score = self._heuristic(next_state)
                        if h_score == float('inf'): continue

                        action_prob = action_log_probs[0, action_idx].item()
                        new_model_log_prob = model_log_prob + action_prob
                        
                        combined_score = new_model_log_prob - (heuristic_weight * h_score)

                        potential_candidates.append((combined_score, new_model_log_prob, path + [base_env.action_map[action_idx]], next_state))

            if not potential_candidates: break

            sorted_candidates = sorted(potential_candidates, key=lambda x: x[0], reverse=True)
            beam = sorted_candidates[:beam_width]

        if beam and check_win(beam[0][3]):
             print(f"✨ Soluzione trovata alla fine della ricerca!")
             return beam[0][2]

        print(f"❌ Nessuna soluzione trovata entro il limite di {max_path_length} passi.")
        return []