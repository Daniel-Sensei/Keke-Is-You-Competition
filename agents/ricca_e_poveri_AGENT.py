import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, List, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces

from baba import (
    GameState, Direction, GameObj, GameObjectType, make_level, parse_map,
    advance_game_state, check_win, character_to_name, name_to_character
)
from base_agent import BaseAgent

# =====================================================================================
# SECTION 1: CUSTOM GYMNASIUM ENVIRONMENT FOR TRAINING
# =====================================================================================

class BabaEnv(gym.Env):
    """
    A custom Gymnasium environment for "Baba Is You".
    This version includes a step-count channel in the observation and a bugfix
    for state hashing.
    """
    def __init__(self, list_of_ascii_maps: list, max_height: int, max_width: int):
        super(BabaEnv, self).__init__()

        self.list_of_ascii_maps = list_of_ascii_maps
        self.max_height = max_height
        self.max_width = max_width

        self.action_space = spaces.Discrete(4)
        self.action_map = {0: Direction.Up, 1: Direction.Down, 2: Direction.Left, 3: Direction.Right}

        # Define object and rule types for observation channels
        self.object_types = list(character_to_name.keys())
        self.num_object_types = len(self.object_types)
        self.rule_types = ['you', 'win', 'push', 'stop', 'kill', 'sink', 'move', 'hot', 'melt']
        self.num_rule_channels = len(self.rule_types)

        # Total channels = object layers + rule layers + 1 step-count layer
        total_channels = self.num_object_types + self.num_rule_channels + 1
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(total_channels, self.max_height, self.max_width),
                                            dtype=np.float32)

        self.current_ascii_map = None
        self.state = None
        self.visited_states = set()
        self.episode_steps = 0
        self.max_episode_steps = 250

    def _pad_map_string(self, ascii_map_str: str) -> str:
        """Pads the ASCII map to the max dimensions."""
        lines = ascii_map_str.split('\n')
        padded_lines = [line.ljust(self.max_width, '_') for line in lines]
        while len(padded_lines) < self.max_height:
            padded_lines.append('_' * self.max_width)
        return "\n".join(padded_lines)

    def _get_obs(self):
        """Constructs the observation tensor from the current game state."""
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if not self.state: return obs

        # Channel for each object type
        all_objects = self.state.phys + self.state.words
        for obj in all_objects:
            char_suffix = "_word" if obj.object_type != GameObjectType.Physical else "_obj"
            char = name_to_character.get(obj.name + char_suffix)
            if char and char in self.object_types:
                channel_idx = self.object_types.index(char)
                if 0 <= obj.y < self.max_height and 0 <= obj.x < self.max_width:
                    obs[channel_idx, obj.y, obj.x] = 1.0

        # Channel for each active rule
        rule_channels_start = self.num_object_types
        if hasattr(self.state, 'rules') and self.state.rules:
            for rule_idx, rule_type in enumerate(self.rule_types):
                for active_rule in self.state.rules:
                    if rule_type in active_rule:
                        obj_name = active_rule.split('-')[0]
                        if obj_name in self.state.sort_phys:
                            for obj in self.state.sort_phys[obj_name]:
                                if 0 <= obj.y < self.max_height and 0 <= obj.x < self.max_width:
                                    obs[rule_channels_start + rule_idx, obj.y, obj.x] = 1.0

        # Final channel for normalized step count
        step_channel_idx = self.num_object_types + self.num_rule_channels
        normalized_steps = self.episode_steps / self.max_episode_steps
        obs[step_channel_idx, :, :] = normalized_steps

        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self.current_ascii_map = random.choice(self.list_of_ascii_maps)
        padded_map_str = self._pad_map_string(self.current_ascii_map)
        game_map_parsed = parse_map(padded_map_str)
        self.state = make_level(game_map_parsed)
        self.visited_states = set()
        return self._get_obs(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("You must call reset() before calling step().")

        self.episode_steps += 1
        prev_state = self.state.copy()
        direction = self.action_map[action]
        self.state = advance_game_state(direction, self.state)

        reward = self._calculate_reward(prev_state, self.state)
        terminated = check_win(self.state) or reward <= -10.0
        truncated = self.episode_steps >= self.max_episode_steps
        observation = self._get_obs()
        info = {'won': check_win(self.state), 'steps': self.episode_steps}

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_state, new_state):
        """Calculates a reward based on game state changes."""
        if check_win(new_state): return 100.0
        if not new_state.players and prev_state.players: return -10.0  # Player disappeared

        reward = -0.01  # Small penalty for each step

        # Reward/penalize rule changes
        if hasattr(prev_state, 'rules') and hasattr(new_state, 'rules'):
            new_rules = set(new_state.rules) - set(prev_state.rules)
            for rule in new_rules:
                if 'you' in rule: reward += 2.0
                elif 'win' in rule: reward += 3.0
                else: reward += 0.5

            lost_rules = set(prev_state.rules) - set(new_state.rules)
            for rule in lost_rules:
                if 'you' in rule and not any('you' in r for r in new_state.rules): reward -= 3.0
                elif 'win' in rule and not any('win' in r for r in new_state.rules): reward -= 2.0

        # Reward for getting closer to a winnable object
        if new_state.players and new_state.winnables:
            try:
                prev_dist = min(abs(p.x - w.x) + abs(p.y - w.y) for p in prev_state.players for w in prev_state.winnables)
                new_dist = min(abs(p.x - w.x) + abs(p.y - w.y) for p in new_state.players for w in new_state.winnables)
                if new_dist < prev_dist: reward += 0.5
            except (ValueError, AttributeError):
                pass

        # Reward for exploring new states
        state_representation = tuple(sorted([str((obj.name, obj.x, obj.y)) for obj in new_state.phys + new_state.words] + new_state.rules))
        state_hash = hash(state_representation)
        if state_hash not in self.visited_states:
            reward += 0.05
            self.visited_states.add(state_hash)

        return reward

# =====================================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURE
# =====================================================================================

class ResidualBlock(nn.Module):
    """A standard Residual Block for a ResNet."""
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
    """A ResNet-based model for processing game state observations."""
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
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.fc_stack(out)

# =====================================================================================
# SECTION 3: MAIN AGENT CLASS
# =====================================================================================

class RICCA_E_POVERIAgent(BaseAgent):
    """
    The main agent, combining a learned policy (ResNet) with heuristic search.
    """
    def __init__(self, model_path='./agents/ricca_e_poveri_model.pth'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.MODEL_PATH = model_path
        self.MODEL_MAX_H, self.MODEL_MAX_W = 20, 20
        self.policy_net = None
        self.optimizer = None
        self.criterion = None

    def initialize_for_training(self, n_channels, height, width, n_actions, learning_rate=1e-4):
        """Initializes the agent's network and optimizer for training."""
        self.policy_net = DQN_ResNet(n_channels, height, width, n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.policy_net.train()
        print("Agent initialized for training with DQN_ResNet.")

    def initialize_for_inference(self):
        """Initializes the agent by loading a pre-trained model for solving levels."""
        dummy_env = BabaEnv([], self.MODEL_MAX_H, self.MODEL_MAX_W)
        total_channels = dummy_env.observation_space.shape[0]
        n_actions = dummy_env.action_space.n
        self.policy_net = DQN_ResNet(total_channels, self.MODEL_MAX_H, self.MODEL_MAX_W, n_actions).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.policy_net.eval()
            print(f"Model loaded successfully from '{self.MODEL_PATH}'.")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at '{self.MODEL_PATH}'.")
            self.policy_net = None

    def save_model(self, path=None):
        """Saves the current model's state dictionary."""
        save_path = path if path else self.MODEL_PATH
        if self.policy_net:
            torch.save(self.policy_net.state_dict(), save_path)
            print(f"Model saved successfully to '{save_path}'.")

    # --- HEURISTIC METHODS ---

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _find_word_objects(self, state: GameState, word_name: str) -> List[GameObj]:
        return [w for w in state.words if w.name == word_name]

    def _get_state_hash(self, state: GameState) -> Optional[str]:
        components = []
        player_pos = sorted([(p.x, p.y) for p in state.players])
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        rules = sorted(state.rules)
        components.append(f"P:{player_pos}")
        components.append(f"W:{word_pos}")
        components.append(f"O:{phys_pos}")
        components.append(f"R:{rules}")
        return "|".join(components)

    def _heuristic(self, state: GameState) -> float:
        if check_win(state): return 0

        # Analyze current rules
        you_rules = any('-is-you' in r for r in state.rules)
        win_rules = any('-is-win' in r for r in state.rules)

        # If there is no player (e.g., 'YOU' rule is broken), calculate the cost
        # to re-form it instead of immediately returning infinity.
        if not state.players or not you_rules:
            cost = self._estimate_rule_formation_cost(state, 'you')
            return 500.0 + cost if cost != float('inf') else float('inf')

        if win_rules:
            if state.winnables:
                # If a win condition exists, the heuristic is the distance to it
                min_dist = min(self._manhattan_distance((p.x, p.y), (w.x, w.y))
                               for p in state.players for w in state.winnables)
                return min_dist
            else:
                # A WIN rule exists, but there are no corresponding winnable objects
                return 100.0

        # If the WIN rule is missing, calculate the cost to create it.
        if not win_rules:
            cost = self._estimate_rule_formation_cost(state, 'win')
            return 100.0 + cost if cost != float('inf') else float('inf')

        return float('inf')  # Fallback case

    def _estimate_rule_formation_cost(self, state: GameState, target_property: str) -> float:
        """Estimates the cost to form a rule like 'NOUN-IS-PROPERTY'."""
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name not in ['is', target_property]]
        is_words = self._find_word_objects(state, 'is')
        property_words = self._find_word_objects(state, target_property)

        if not nouns or not is_words or not property_words:
            return float('inf')

        min_formation_dist = float('inf')
        for noun in nouns:
            dist_to_is = min(self._manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words)
            dist_to_prop = min(self._manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words)
            min_formation_dist = min(min_formation_dist, dist_to_is + dist_to_prop)

        return min_formation_dist if min_formation_dist != float('inf') else float('inf')

    # --- BEAM SEARCH ---

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        """
        Performs a beam search guided by both the learned policy and the heuristic.
        """
        if self.policy_net is None: self.initialize_for_inference()
        if self.policy_net is None: return None

        base_env = BabaEnv(["\n".join("".join(row) for row in initial_state.orig_map)], self.MODEL_MAX_H, self.MODEL_MAX_W)
        base_env.reset()

        n_actions = base_env.action_space.n
        beam_width = 20
        max_path_length = 100
        heuristic_weight = 1.5  # Balance between network_policy and heuristic

        # Each item in the beam: (combined_score, model_log_prob, path, game_state)
        beam = [(0.0, 0.0, [], initial_state.copy())]
        visited_states = {} # Tracks visited states to avoid redundant exploration

        for step in range(max_path_length):
            if not beam: break

            potential_candidates = []
            for score, model_log_prob, path, game_state in beam:
                if check_win(game_state):
                    print(f"Solution found with path length {len(path)}!")
                    return path

                # Use the robust state hashing method
                state_hash = self._get_state_hash(game_state)
                if state_hash in visited_states and visited_states[state_hash] >= model_log_prob:
                    continue # Prune if a better path to this state has been found
                visited_states[state_hash] = model_log_prob

                # Get the observation for the current state
                temp_env = BabaEnv([base_env.current_ascii_map], self.MODEL_MAX_H, self.MODEL_MAX_W)
                temp_env.state = game_state
                temp_env.episode_steps = len(path)
                obs = temp_env._get_obs()

                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32).unsqueeze(0)
                    action_log_probs = F.log_softmax(self.policy_net(state_tensor), dim=1)

                    for action_idx in range(n_actions):
                        next_state = advance_game_state(base_env.action_map[action_idx], game_state.copy())

                        h_score = self._heuristic(next_state)
                        if h_score == float('inf'): continue # Prune dead-end states

                        action_prob = action_log_probs[0, action_idx].item()
                        new_model_log_prob = model_log_prob + action_prob
                        
                        # Score combines the model's confidence and the heuristic's guidance
                        combined_score = new_model_log_prob - (heuristic_weight * h_score)

                        potential_candidates.append((combined_score, new_model_log_prob, path + [base_env.action_map[action_idx]], next_state))

            if not potential_candidates: break

            # Select the top candidates for the next beam
            sorted_candidates = sorted(potential_candidates, key=lambda x: x[0], reverse=True)
            beam = sorted_candidates[:beam_width]

        # Final check if a solution was found in the last step
        if beam and check_win(beam[0][3]):
             print(f"Solution found at the end of the search!")
             return beam[0][2]

        print(f"No solution found within the {max_path_length} step limit.")
        return []