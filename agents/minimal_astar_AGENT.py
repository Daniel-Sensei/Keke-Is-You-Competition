# agents/minimal_astar_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq


class MinimalASTARAgent(BaseAgent):
    PENALTY_NO_PLAYER_CONTROL = 30.0
    PENALTY_NO_WIN_CONDITION_RULE = 30.0
    PENALTY_NO_WINNABLE_OBJECTS = 20.0
    PENALTY_NO_PLAYER_OBJECTS = 20.0
    PENALTY_WORD_TYPE_MISSING = 10.0

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
                    # Horizontal
                    h1 = abs(t1.x - (ix - 1)) + abs(t1.y - iy)
                    h3 = abs(t3.x - (ix + 1)) + abs(t3.y - iy)
                    # Vertical
                    v1 = abs(t1.x - ix) + abs(t1.y - (iy - 1))
                    v3 = abs(t3.x - ix) + abs(t3.y - (iy + 1))
                    min_cost = min(min_cost, h1 + h3, v1 + v3)
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

        return min(abs(p.x - w.x) + abs(p.y - w.y)
                   for p in state.players for w in state.winnables)

    def _heuristic_rules(self, state: GameState) -> float:
        rules = set(state.rules)
        cost = 0.0

        if not any("you" in r for r in rules):
            possible_you_rules = self._get_viable_you_rules(state)
            est_costs = [self._estimate_rule_formation_cost(state, x, "is", "you") for x, _, _ in possible_you_rules]
            cost += min(est_costs) if est_costs else self.PENALTY_NO_PLAYER_CONTROL
        elif not state.players:
            cost += self.PENALTY_NO_PLAYER_OBJECTS

        if not any("win" in r for r in rules):
            cost += self._estimate_rule_formation_cost(state, "flag", "is", "win")
        elif not state.winnables:
            cost += self.PENALTY_NO_WINNABLE_OBJECTS

        return cost

    def _calculate_heuristic(self, state: GameState) -> float:
        hkey = self._get_state_hash(state)
        if hkey in self.heuristic_cache:
            return self.heuristic_cache[hkey]

        num_rules = len(state.rules)
        w_rules = 2.0 if num_rules < 2 else 0.8
        w_dist = 0.5 if num_rules < 2 else 1.2

        rule_cost = self._heuristic_rules(state)
        dist_cost = self._heuristic_distance(state)

        # Penalizza stati terminali (KILL, SINK)
        for player in state.players:
            if player in state.killers or player in state.sinkers:
                return float('inf')

        total = rule_cost * w_rules + dist_cost * w_dist
        self.heuristic_cache[hkey] = total
        return total

    def search(self, initial_state: GameState, iterations: int = 250) -> List[Direction]:
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}

        init_hash = self._get_state_hash(initial_state)
        g_costs[init_hash] = 0.0
        h = self._calculate_heuristic(initial_state)
        heapq.heappush(open_set, (h, 0.0, init_hash, initial_state, []))

        while open_set and iterations > 0:
            f, g, hash_key, state, path = heapq.heappop(open_set)
            if hash_key in closed_set: continue
            closed_set.add(hash_key)

            if check_win(state):
                return path

            for action in self.possible_actions:
                new_state = advance_game_state(action, state.copy())
                new_hash = self._get_state_hash(new_state)
                new_g = g + 1.0
                if new_g >= g_costs.get(new_hash, float('inf')):
                    continue

                g_costs[new_hash] = new_g
                h = self._calculate_heuristic(new_state)
                if h == float('inf'): continue

                new_path = path + [action]
                heapq.heappush(open_set, (new_g + h, new_g, new_hash, new_state, new_path))

            iterations -= 1

        return []
