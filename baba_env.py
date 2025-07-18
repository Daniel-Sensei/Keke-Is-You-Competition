# baba_env.py (Versione DEFINITIVA con Canale Step-Count e Bugfix)

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
        
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: Direction.Up, 1: Direction.Down, 2: Direction.Left, 3: Direction.Right}
        
        self.object_types = list(character_to_name.keys())
        self.num_object_types = len(self.object_types)
        
        self.rule_types = ['you', 'win', 'push', 'stop', 'kill', 'sink', 'move', 'hot', 'melt']
        self.num_rule_channels = len(self.rule_types)
        
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
        lines = ascii_map_str.split('\n')
        padded_lines = [line.ljust(self.max_width, '_') for line in lines]
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
        if self.state is None: raise RuntimeError("Devi chiamare reset() prima di step()")

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
        if check_win(new_state): return 100.0
        if not new_state.players and prev_state.players: return -10.0
        reward = -0.01
        
        if hasattr(prev_state, 'rules') and hasattr(new_state, 'rules'):
            new_rules_formed = set(new_state.rules) - set(prev_state.rules)
            for rule in new_rules_formed:
                if 'you' in rule: reward += 2.0
                elif 'win' in rule: reward += 3.0
                elif 'push' in rule: reward += 1.0
                elif 'stop' in rule or 'kill' in rule: reward += 0.5
            
            lost_rules = set(prev_state.rules) - set(new_state.rules)
            for rule in lost_rules:
                if 'you' in rule and not any('you' in r for r in new_state.rules): reward -= 3.0
                elif 'win' in rule and not any('win' in r for r in new_state.rules): reward -= 2.0
        
        if new_state.players and new_state.winnables and prev_state.players and prev_state.winnables:
            try:
                prev_dist = min(abs(p.x - w.x) + abs(p.y - w.y) for p in prev_state.players for w in prev_state.winnables)
                new_dist = min(abs(p.x - w.x) + abs(p.y - w.y) for p in new_state.players for w in new_state.winnables)
                if new_dist < prev_dist: reward += 0.5
                elif new_dist > prev_dist: reward -= 0.1
            except ValueError: pass
        
        # ---> RIGA CORRETTA <---
        state_representation = tuple(sorted([str((obj.name, obj.x, obj.y)) for obj in new_state.phys + new_state.words] + new_state.rules))

        state_hash = hash(state_representation)
        if state_hash not in self.visited_states:
            reward += 0.05
            self.visited_states.add(state_hash)
        
        return reward