# agents/advanced_astar_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq


class AdvancedASTARAgent(BaseAgent):
    PENALTY_NO_PLAYER_CONTROL = 25.0  # Reduced for better exploration
    PENALTY_NO_WIN_CONDITION_RULE = 25.0
    PENALTY_NO_WINNABLE_OBJECTS = 15.0
    PENALTY_NO_PLAYER_OBJECTS = 15.0
    PENALTY_WORD_TYPE_MISSING = 8.0

    def __init__(self):
        self.heuristic_cache = {}
        self.possible_actions = [d for d in Direction if d != Direction.Undefined]

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
        """Get all possible win rule formations"""
        is_words = self._get_word_objects_by_base_name(state, "is")
        win_words = self._get_word_objects_by_base_name(state, "win")
        rules = []
        
        # Check for all possible objects that could become win
        possible_win_objects = ["flag", "baba", "skull", "rock", "keke", "floor", "wall"]
        for obj in possible_win_objects:
            if self._get_word_objects_by_base_name(state, obj):
                rules.append((obj, "is", "win"))
        return rules if is_words and win_words else []

    def _estimate_rule_formation_cost(self, state: GameState, term1_base: str, term2_base_is: str, term3_base: str) -> float:
        term1_objs = self._get_word_objects_by_base_name(state, term1_base)
        is_objs = self._get_word_objects_by_base_name(state, term2_base_is)
        term3_objs = self._get_word_objects_by_base_name(state, term3_base)

        if not term1_objs or not is_objs or not term3_objs:
            return self.PENALTY_WORD_TYPE_MISSING

        min_cost = float('inf')
        for is_obj in is_objs:
            ix, iy = is_obj.x, is_obj.y
            for t1 in term1_objs:
                for t3 in term3_objs:
                    # Horizontal formation
                    h1 = abs(t1.x - (ix - 1)) + abs(t1.y - iy)
                    h3 = abs(t3.x - (ix + 1)) + abs(t3.y - iy)
                    horizontal_cost = h1 + h3
                    
                    # Vertical formation  
                    v1 = abs(t1.x - ix) + abs(t1.y - (iy - 1))
                    v3 = abs(t3.x - ix) + abs(t3.y - (iy + 1))
                    vertical_cost = v1 + v3
                    
                    formation_cost = min(horizontal_cost, vertical_cost)
                    
                    # Bonus for very close formations
                    if formation_cost <= 2:
                        formation_cost *= 0.6
                    elif formation_cost <= 4:
                        formation_cost *= 0.8
                    
                    min_cost = min(min_cost, formation_cost)
        return min_cost

    def _get_state_hash(self, game_state: GameState) -> tuple:
        phys = tuple(sorted((p.name, p.x, p.y) for p in game_state.phys if hasattr(p, 'name')))
        words = tuple(sorted((w.name, w.x, w.y) for w in (game_state.words + game_state.keywords)))
        rules = tuple(sorted(set(game_state.rules)))
        return (phys, words, rules)

    def _heuristic_distance(self, state: GameState) -> float:
        if check_win(state): return 0.0
        if not state.players: return self.PENALTY_NO_PLAYER_OBJECTS
        if not any("win" in r for r in state.rules): return 0.0
        if not state.winnables: return self.PENALTY_NO_WINNABLE_OBJECTS

        # Enhanced distance calculation for complex levels
        distances = []
        for p in state.players:
            for w in state.winnables:
                dist = abs(p.x - w.x) + abs(p.y - w.y)
                distances.append(dist)
        
        if not distances:
            return self.PENALTY_NO_WINNABLE_OBJECTS
            
        min_dist = min(distances)
        
        # Apply distance scaling for very long distances
        if min_dist > 15:
            return min_dist * 0.7  # Reduce penalty for very long distances
        elif min_dist > 8:
            return min_dist * 0.85
        else:
            return min_dist

    def _heuristic_rules(self, state: GameState) -> float:
        rules = set(state.rules)
        cost = 0.0

        # Check for "you" rules
        if not any("you" in r for r in rules):
            possible_you_rules = self._get_viable_you_rules(state)
            est_costs = [self._estimate_rule_formation_cost(state, x, "is", "you") for x, _, _ in possible_you_rules]
            cost += min(est_costs) if est_costs else self.PENALTY_NO_PLAYER_CONTROL
        elif not state.players:
            cost += self.PENALTY_NO_PLAYER_OBJECTS

        # Enhanced win rule checking
        if not any("win" in r for r in rules):
            possible_win_rules = self._get_viable_win_rules(state)
            if possible_win_rules:
                win_costs = [self._estimate_rule_formation_cost(state, x, "is", "win") for x, _, _ in possible_win_rules]
                min_win_cost = min(win_costs)
                
                # Special bonus for specific level patterns
                # Level 6 pattern: baba-is-you already exists, need something-is-win
                if any("baba-is-you" in r for r in rules):
                    min_win_cost *= 0.8  # Prioritize win formation when baba-is-you exists
                
                cost += min_win_cost
            else:
                cost += self.PENALTY_NO_WIN_CONDITION_RULE
        elif not state.winnables:
            cost += self.PENALTY_NO_WINNABLE_OBJECTS

        return cost

    def _calculate_heuristic(self, state: GameState) -> float:
        hkey = self._get_state_hash(state)
        if hkey in self.heuristic_cache:
            return self.heuristic_cache[hkey]

        num_rules = len(state.rules)
        
        # Adaptive weighting based on game state complexity
        if num_rules < 2:
            # Early game: focus heavily on rule formation
            w_rules = 1.8
            w_dist = 0.4
        elif not state.winnables:
            # Need win objects: balance rule and movement
            w_rules = 1.2
            w_dist = 0.6
        else:
            # End game: focus on reaching win objects with some rule awareness
            w_rules = 0.7
            w_dist = 1.0

        rule_cost = self._heuristic_rules(state)
        dist_cost = self._heuristic_distance(state)

        # Enhanced terminal state checking
        for player in state.players:
            if player in state.killers or player in state.sinkers:
                return float('inf')

        total = rule_cost * w_rules + dist_cost * w_dist
        self.heuristic_cache[hkey] = total
        return total

    def search(self, initial_state: GameState, iterations: int = 800) -> List[Direction]:
        """Enhanced A* search with increased exploration capacity"""
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}

        init_hash = self._get_state_hash(initial_state)
        g_costs[init_hash] = 0.0
        h = self._calculate_heuristic(initial_state)
        heapq.heappush(open_set, (h, 0.0, init_hash, initial_state, []))

        max_path_length = 50  # Increased for complex levels
        max_open_set_size = 5000  # Better memory management
        nodes_explored = 0
        
        while open_set and iterations > 0:
            f, g, hash_key, state, path = heapq.heappop(open_set)
            if hash_key in closed_set: 
                continue
            closed_set.add(hash_key)
            nodes_explored += 1

            if check_win(state):
                return path
                
            # Dynamic path length limit based on exploration progress
            current_max_length = min(max_path_length, 30 + nodes_explored // 100)
            if len(path) >= current_max_length:
                continue

            for action in self.possible_actions:
                new_state = advance_game_state(action, state.copy())
                new_hash = self._get_state_hash(new_state)
                new_g = g + 1.0
                
                if new_g >= g_costs.get(new_hash, float('inf')):
                    continue

                g_costs[new_hash] = new_g
                h = self._calculate_heuristic(new_state)
                if h == float('inf'): 
                    continue

                new_path = path + [action]
                heapq.heappush(open_set, (new_g + h, new_g, new_hash, new_state, new_path))

            # Memory management with gradual pruning
            if len(open_set) > max_open_set_size:
                # Keep the best 60% of states
                keep_size = int(max_open_set_size * 0.6)
                open_set = heapq.nsmallest(keep_size, open_set)
                heapq.heapify(open_set)

            iterations -= 1

        return []
