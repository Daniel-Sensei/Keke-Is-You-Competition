# baba_env.py (Versione Migliorata con Rappresentazione delle Regole)

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
        
        # Canali per oggetti + canali per regole importanti
        self.object_types = list(character_to_name.keys())
        self.num_object_types = len(self.object_types)
        
        # Aggiungi canali per le regole principali
        self.rule_types = ['you', 'win', 'push', 'stop', 'kill', 'sink', 'move', 'hot', 'melt']
        self.num_rule_channels = len(self.rule_types)
        
        # Totale canali = oggetti + regole
        total_channels = self.num_object_types + self.num_rule_channels
        
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(total_channels, self.max_height, self.max_width), 
                                            dtype=np.float32)
        
        self.current_ascii_map = None
        self.state = None
        self.visited_states = set()
        self.episode_steps = 0
        self.max_episode_steps = 200

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
        if not self.state: 
            return obs

        # Canali per oggetti fisici e parole
        all_objects = self.state.phys + self.state.words
        for obj in all_objects:
            char_suffix = "_word" if obj.object_type != GameObjectType.Physical else "_obj"
            char = name_to_character.get(obj.name + char_suffix)
            
            if char and char in self.object_types:
                channel_idx = self.object_types.index(char)
                if 0 <= obj.y < self.max_height and 0 <= obj.x < self.max_width:
                    obs[channel_idx, obj.y, obj.x] = 1.0
        
        # NUOVO: Canali per le regole attive
        rule_channels_start = self.num_object_types
        
        if hasattr(self.state, 'rules') and self.state.rules:
            for rule_idx, rule_type in enumerate(self.rule_types):
                for active_rule in self.state.rules:
                    if rule_type in active_rule:
                        # Estrai il nome dell'oggetto dalla regola
                        obj_name = active_rule.split('-')[0]
                        
                        # Marca tutti gli oggetti di quel tipo
                        if obj_name in self.state.sort_phys:
                            for obj in self.state.sort_phys[obj_name]:
                                if 0 <= obj.y < self.max_height and 0 <= obj.x < self.max_width:
                                    obs[rule_channels_start + rule_idx, obj.y, obj.x] = 1.0
        
        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Scegli un livello a caso
        self.current_ascii_map = random.choice(self.list_of_ascii_maps)
        padded_map_str = self._pad_map_string(self.current_ascii_map)
        
        game_map_parsed = parse_map(padded_map_str)
        self.state = make_level(game_map_parsed)
        
        # Reset contatori
        self.visited_states = set()
        self.episode_steps = 0
        
        return self._get_obs(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Devi chiamare reset() prima di step()")

        self.episode_steps += 1
        
        prev_state = self.state.copy()
        direction = self.action_map[action]
        self.state = advance_game_state(direction, self.state)
        
        reward = self._calculate_reward(prev_state, self.state)
        
        # Termina se vinto, morto o troppi step
        terminated = check_win(self.state) or reward <= -10.0
        truncated = self.episode_steps >= self.max_episode_steps
        
        observation = self._get_obs()
        info = {'won': check_win(self.state), 'steps': self.episode_steps}
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_state, new_state):
        # Vittoria
        if check_win(new_state): 
            return 100.0
        
        # Morte del player
        if not new_state.players and prev_state.players: 
            return -10.0
        
        # Penalità base per step
        reward = -0.01
        
        # Reward per formazione di nuove regole utili
        if hasattr(prev_state, 'rules') and hasattr(new_state, 'rules'):
            new_rules_formed = set(new_state.rules) - set(prev_state.rules)
            for rule in new_rules_formed:
                if 'you' in rule:
                    reward += 2.0  # Molto importante mantenere il controllo
                elif 'win' in rule:
                    reward += 3.0  # Critico per vincere
                elif 'push' in rule:
                    reward += 1.0
                elif 'stop' in rule or 'kill' in rule:
                    reward += 0.5
            
            # Penalità per distruggere regole importanti
            lost_rules = set(prev_state.rules) - set(new_state.rules)
            for rule in lost_rules:
                if 'you' in rule and not any('you' in r for r in new_state.rules):
                    reward -= 3.0  # Grave se perdiamo TUTTI i controlli
                elif 'win' in rule and not any('win' in r for r in new_state.rules):
                    reward -= 2.0  # Male se non c'è più modo di vincere
        
        # Reward basato sulla distanza player-win
        if new_state.players and new_state.winnables and prev_state.players and prev_state.winnables:
            try:
                prev_dist = min(manhattan_distance((p.x, p.y), (w.x, w.y)) 
                               for p in prev_state.players for w in prev_state.winnables)
                new_dist = min(manhattan_distance((p.x, p.y), (w.x, w.y)) 
                              for p in new_state.players for w in new_state.winnables)
                
                # Reward più significativo per avvicinarsi
                if new_dist < prev_dist: 
                    reward += 0.5
                elif new_dist > prev_dist:
                    reward -= 0.1
                    
            except ValueError: 
                pass
        
        # Bonus per esplorare nuove configurazioni
        # Crea un hash dello stato basato sulle posizioni degli oggetti
        state_representation = []
        for obj in new_state.phys + new_state.words:
            state_representation.append((obj.name, obj.x, obj.y))
        # Aggiungi anche le regole attive
        state_representation.extend(sorted(new_state.rules))
        
        state_hash = hash(tuple(state_representation))
        if state_hash not in self.visited_states:
            reward += 0.05
            self.visited_states.add(state_hash)
        
        return reward

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])