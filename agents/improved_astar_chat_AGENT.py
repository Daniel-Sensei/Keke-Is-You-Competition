# agents/improved_astar_chat_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq


class ImprovedASTAR_CHATAgent(BaseAgent):
    # Penalità ridotte per permettere più esplorazione
    PENALTY_NO_PLAYER_CONTROL = 25.0
    PENALTY_NO_WIN_CONDITION_RULE = 25.0
    PENALTY_NO_WINNABLE_OBJECTS = 15.0
    PENALTY_NO_PLAYER_OBJECTS = 15.0
    PENALTY_WORD_TYPE_MISSING = 8.0
    
    # Nuove penalità per situazioni specifiche
    PENALTY_PLAYER_IN_DANGER = 50.0
    BONUS_RULE_FORMATION_PROGRESS = -5.0

    def __init__(self):
        self.heuristic_cache = {}
        self.possible_actions = [d for d in Direction if d != Direction.Undefined]
        self.rule_formation_cache = {}

    def _get_word_objects_by_base_name(self, state: GameState, base_name: str) -> List[GameObj]:
        return [obj for lst in [state.words, state.keywords] if lst for obj in lst if hasattr(obj, 'name') and obj.name == base_name]

    def _get_viable_you_rules(self, state: GameState):
        is_words = self._get_word_objects_by_base_name(state, "is")
        you_words = self._get_word_objects_by_base_name(state, "you")
        rules = []
        for x in {w.name for w in state.words if w.name != "is"}:
            if self._get_word_objects_by_base_name(state, x):
                rules.append((x, "is", "you"))
        return rules if is_words and you_words else []

    def _get_viable_win_rules(self, state: GameState):
        """Find viable win rule formations"""
        is_words = self._get_word_objects_by_base_name(state, "is")
        win_words = self._get_word_objects_by_base_name(state, "win")
        rules = []
        
        # Objects that could become win
        for x in {w.name for w in state.words if w.name not in ["is", "win"]}:
            if self._get_word_objects_by_base_name(state, x):
                rules.append((x, "is", "win"))
        return rules if is_words and win_words else []

    def _estimate_rule_formation_cost(self, state: GameState, term1_base: str, term2_base_is: str, term3_base: str) -> float:
        """Improved rule formation cost calculation"""
        cache_key = (term1_base, term2_base_is, term3_base)
        if cache_key in self.rule_formation_cache:
            return self.rule_formation_cache[cache_key]
            
        term1_objs = self._get_word_objects_by_base_name(state, term1_base)
        is_objs = self._get_word_objects_by_base_name(state, term2_base_is)
        term3_objs = self._get_word_objects_by_base_name(state, term3_base)

        if not term1_objs or not is_objs or not term3_objs:
            cost = self.PENALTY_WORD_TYPE_MISSING
            self.rule_formation_cache[cache_key] = cost
            return cost

        min_cost = float('inf')
        
        for is_obj in is_objs:
            ix, iy = is_obj.x, is_obj.y
            for t1 in term1_objs:
                for t3 in term3_objs:
                    # Horizontal formation: T1 IS T3
                    h1 = abs(t1.x - (ix - 1)) + abs(t1.y - iy)
                    h3 = abs(t3.x - (ix + 1)) + abs(t3.y - iy)
                    horizontal_cost = h1 + h3
                    
                    # Vertical formation
                    v1 = abs(t1.x - ix) + abs(t1.y - (iy - 1))
                    v3 = abs(t3.x - ix) + abs(t3.y - (iy + 1))
                    vertical_cost = v1 + v3
                    
                    # Take minimum and apply discount for closer formations
                    formation_cost = min(horizontal_cost, vertical_cost)
                    if formation_cost <= 2:  # Very close formations get bonus
                        formation_cost *= 0.5
                    elif formation_cost <= 4:  # Close formations get small bonus
                        formation_cost *= 0.8
                    
                    min_cost = min(min_cost, formation_cost)
        
        self.rule_formation_cache[cache_key] = min_cost
        return min_cost

    def _check_player_safety(self, state: GameState) -> float:
        """Check if players are in immediate danger"""
        penalty = 0.0
        
        if not state.players:
            return 0.0
            
        for player in state.players:
            # Check if player is on dangerous tiles
            if hasattr(player, 'x') and hasattr(player, 'y'):
                for killer in (state.killers or []):
                    if hasattr(killer, 'x') and hasattr(killer, 'y'):
                        if player.x == killer.x and player.y == killer.y:
                            penalty += self.PENALTY_PLAYER_IN_DANGER
                
                for sinker in (state.sinkers or []):
                    if hasattr(sinker, 'x') and hasattr(sinker, 'y'):
                        if player.x == sinker.x and player.y == sinker.y:
                            penalty += self.PENALTY_PLAYER_IN_DANGER
        
        return penalty

    def _get_state_hash(self, game_state: GameState) -> tuple:
        phys = tuple(sorted((p.name, p.x, p.y) for p in game_state.phys if hasattr(p, 'name')))
        words = tuple(sorted((w.name, w.x, w.y) for w in (game_state.words + game_state.keywords)))
        rules = tuple(sorted(set(game_state.rules)))
        return (phys, words, rules)

    def _heuristic_distance(self, state: GameState) -> float:
        """Improved distance heuristic"""
        if check_win(state): 
            return 0.0
        if not state.players: 
            return self.PENALTY_NO_PLAYER_OBJECTS
        if not any("win" in r for r in state.rules): 
            return 0.0
        if not state.winnables: 
            return self.PENALTY_NO_WINNABLE_OBJECTS

        # Calculate minimum distance with improved pathfinding consideration
        min_distance = float('inf')
        
        for player in state.players:
            for winnable in state.winnables:
                distance = abs(player.x - winnable.x) + abs(player.y - winnable.y)
                
                # Apply discount for very close winnables
                if distance <= 2:
                    distance *= 0.7
                elif distance <= 4:
                    distance *= 0.9
                
                min_distance = min(min_distance, distance)
        
        return min_distance

    def _heuristic_rules(self, state: GameState) -> float:
        """Improved rule formation heuristic"""
        rules = set(state.rules)
        cost = 0.0

        # Check for "you" rules
        has_you_rule = any("you" in r for r in rules)
        if not has_you_rule:
            possible_you_rules = self._get_viable_you_rules(state)
            if possible_you_rules:
                est_costs = [self._estimate_rule_formation_cost(state, x, "is", "you") for x, _, _ in possible_you_rules]
                min_you_cost = min(est_costs) if est_costs else self.PENALTY_NO_PLAYER_CONTROL
                cost += min_you_cost
            else:
                cost += self.PENALTY_NO_PLAYER_CONTROL
        elif not state.players:
            cost += self.PENALTY_NO_PLAYER_OBJECTS

        # Check for "win" rules
        has_win_rule = any("win" in r for r in rules)
        if not has_win_rule:
            possible_win_rules = self._get_viable_win_rules(state)
            if possible_win_rules:
                est_costs = [self._estimate_rule_formation_cost(state, x, "is", "win") for x, _, _ in possible_win_rules]
                min_win_cost = min(est_costs) if est_costs else self.PENALTY_NO_WIN_CONDITION_RULE
                cost += min_win_cost
            else:
                cost += self.PENALTY_NO_WIN_CONDITION_RULE
        elif not state.winnables:
            cost += self.PENALTY_NO_WINNABLE_OBJECTS

        return cost

    def _calculate_heuristic(self, state: GameState) -> float:
        """Improved heuristic calculation with better balancing"""
        hkey = self._get_state_hash(state)
        if hkey in self.heuristic_cache:
            return self.heuristic_cache[hkey]

        # Check for terminal losing states first
        safety_penalty = self._check_player_safety(state)
        if safety_penalty > self.PENALTY_PLAYER_IN_DANGER * 0.5:
            return float('inf')

        num_rules = len(state.rules)
        
        # Adaptive weights based on game state
        if num_rules < 2:
            # Early game: focus on rule formation
            w_rules = 1.5
            w_dist = 0.3
        elif not state.winnables:
            # Need win objects: still focus on rules
            w_rules = 1.2
            w_dist = 0.5
        else:
            # End game: focus on reaching win objects
            w_rules = 0.6
            w_dist = 1.0

        rule_cost = self._heuristic_rules(state)
        dist_cost = self._heuristic_distance(state)

        # Bonus for making progress towards rule formation
        bonus = 0.0
        if rule_cost > 0 and rule_cost < 10:  # Making progress on rules
            bonus += self.BONUS_RULE_FORMATION_PROGRESS

        total = rule_cost * w_rules + dist_cost * w_dist + safety_penalty + bonus
        self.heuristic_cache[hkey] = total
        return total

    def search(self, initial_state: GameState, iterations: int = 300) -> List[Direction]:
        """Improved A* search with better exploration"""
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}

        init_hash = self._get_state_hash(initial_state)
        g_costs[init_hash] = 0.0
        h = self._calculate_heuristic(initial_state)
        heapq.heappush(open_set, (h, 0.0, init_hash, initial_state, []))

        nodes_explored = 0
        max_path_length = 50  # Prevent infinite loops
        
        while open_set and iterations > 0:
            f, g, hash_key, state, path = heapq.heappop(open_set)
            
            if hash_key in closed_set: 
                continue
            closed_set.add(hash_key)
            nodes_explored += 1

            if check_win(state):
                return path

            # Prevent overly long paths
            if len(path) >= max_path_length:
                continue

            for action in self.possible_actions:
                new_state = advance_game_state(action, state.copy())
                new_hash = self._get_state_hash(new_state)
                new_g = g + 1.0
                
                # Improved duplicate detection
                if new_g >= g_costs.get(new_hash, float('inf')):
                    continue

                g_costs[new_hash] = new_g
                h = self._calculate_heuristic(new_state)
                
                # Skip obviously bad states
                if h == float('inf'): 
                    continue

                new_path = path + [action]
                heapq.heappush(open_set, (new_g + h, new_g, new_hash, new_state, new_path))

            iterations -= 1

        return []
