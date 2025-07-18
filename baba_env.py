# baba_env.py (Versione per Addestramento Multi-livello)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from baba import GameState, Direction, GameObjectType, make_level, parse_map, advance_game_state, check_win, character_to_name, name_to_character

class BabaEnv(gym.Env):
    def __init__(self, list_of_ascii_maps: list, max_height: int, max_width: int):
        super(BabaEnv, self).__init__()
        
        self.list_of_ascii_maps = list_of_ascii_maps
        self.max_height = max_height
        self.max_width = max_width
        
        # Le azioni rimangono le stesse
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: Direction.Up, 1: Direction.Down, 2: Direction.Left, 3: Direction.Right}
        
        # Lo spazio degli stati ora usa le dimensioni massime
        self.object_types = list(character_to_name.keys())
        self.num_object_types = len(self.object_types)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(self.num_object_types, self.max_height, self.max_width), 
                                            dtype=np.float32)
        
        self.current_ascii_map = None
        self.state = None

    def _pad_map_string(self, ascii_map_str: str) -> str:
        """Aggiunge padding a una mappa stringa per raggiungere le dimensioni massime."""
        lines = ascii_map_str.split('\n')
        padded_lines = []
        
        # Padding orizzontale
        for line in lines:
            padded_line = line.ljust(self.max_width, '_')
            padded_lines.append(padded_line)
            
        # Padding verticale
        while len(padded_lines) < self.max_height:
            padded_lines.append('_' * self.max_width)
            
        return "\n".join(padded_lines)

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if not self.state: return obs

        all_objects = self.state.phys + self.state.words
        for obj in all_objects:
            char_suffix = "_word" if obj.object_type != GameObjectType.Physical else "_obj"
            char = name_to_character.get(obj.name + char_suffix)
            
            if char and char in self.object_types:
                channel_idx = self.object_types.index(char)
                if 0 <= obj.y < self.max_height and 0 <= obj.x < self.max_width:
                    obs[channel_idx, obj.y, obj.x] = 1.0
        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        # ✅ SCEGLI UN LIVELLO A CASO E IMPOSTALO
        self.current_ascii_map = random.choice(self.list_of_ascii_maps)
        padded_map_str = self._pad_map_string(self.current_ascii_map)
        
        game_map_parsed = parse_map(padded_map_str)
        self.state = make_level(game_map_parsed)
        
        return self._get_obs(), {}

    def step(self, action):
        # ... (il metodo step rimane quasi identico a prima) ...
        # Assicurati solo che gestisca correttamente il caso in cui self.state non è inizializzato
        if self.state is None:
            raise RuntimeError("Devi chiamare reset() prima di step()")

        prev_state = self.state.copy()
        direction = self.action_map[action]
        self.state = advance_game_state(direction, self.state)
        reward = self._calculate_reward(prev_state, self.state)
        terminated = check_win(self.state) or reward <= -1.0
        truncated = False
        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_state, new_state):
        # ... (il metodo _calculate_reward rimane identico a prima) ...
        if check_win(new_state): return 10.0
        if not new_state.players and prev_state.players: return -1.0
        reward = -0.01
        new_rules_formed = set(new_state.rules) - set(prev_state.rules)
        for rule in new_rules_formed:
            if 'push' in rule or 'win' in rule or 'you' in rule: reward += 1.0
        if new_state.players and new_state.winnables and prev_state.players and prev_state.winnables:
            try:
                prev_dist = min(manhattan_distance((p.x, p.y), (w.x, w.y)) for p in prev_state.players for w in prev_state.winnables)
                new_dist = min(manhattan_distance((p.x, p.y), (w.x, w.y)) for p in new_state.players for w in new_state.winnables)
                if new_dist < prev_dist: reward += 0.1
            except ValueError: pass
        return reward

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])