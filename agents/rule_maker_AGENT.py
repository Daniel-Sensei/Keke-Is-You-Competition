import heapq
import itertools
import time
from collections import deque
from typing import List, Dict, Tuple, Optional

from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win, name_to_character, make_level, parse_map, double_map_to_string)

# ... (Funzioni ausiliarie e classi di supporto sono invariate) ...
def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int: return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
def find_word_objects(state: GameState, word_name: str) -> List[GameObj]: return [w for w in state.words if w.name == word_name]
def apply_solution(initial_state: GameState, actions: List[Direction]) -> GameState:
    current_state = initial_state.copy()
    for move in actions: current_state = advance_game_state(move, current_state)
    return current_state

class HeapQEntry:
    def __init__(self, priority: float, g_score: int, tie_breaker: int, actions: List[Direction], state: GameState):
        self.priority, self.g_score, self.tie_breaker, self.actions, self.state = priority, g_score, tie_breaker, actions, state
    def __lt__(self, other):
        if self.priority != other.priority: return self.priority < other.priority
        if self.g_score != other.g_score: return self.g_score > other.g_score
        return self.tie_breaker < other.tie_breaker

# --- Agenti Interni Specializzati ---
class _BaseAStarAgent:
    def __init__(self):
        self.counter = itertools.count()
        self.max_path_length = 120
        self.PUSH_COST = 10 

    def _get_state_hash(self, state: GameState) -> str: return "|".join([f"P:{sorted([(p.x, p.y) for p in state.players])}", f"W:{sorted([(w.name, w.x, w.y) for w in state.words])}", f"O:{sorted([(o.name, o.x, o.y) for o in state.phys])}"])

    def _get_push_aware_distance(self, state: GameState, start_pos: Tuple[int, int], win_positions: set) -> float:
        q = [(manhattan_distance(start_pos, min(win_positions, key=lambda p: manhattan_distance(start_pos, p))), 0, start_pos)]
        visited = {start_pos: 0}
        w, h = len(state.obj_map[0]), len(state.obj_map)
        while q:
            f, g, pos = heapq.heappop(q)
            if pos in win_positions: return g
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h): continue
                obj = state.obj_map[next_pos[1]][next_pos[0]]
                move_cost = 1
                if isinstance(obj, GameObj):
                    if obj in state.pushables: move_cost = self.PUSH_COST
                    elif obj.object_type == GameObjectType.Physical: continue
                new_g = g + move_cost
                if new_g < visited.get(next_pos, float('inf')):
                    visited[next_pos] = new_g
                    new_f = new_g + manhattan_distance(next_pos, min(win_positions, key=lambda p: manhattan_distance(next_pos, p)))
                    heapq.heappush(q, (new_f, new_g, next_pos))
        return float('inf')

    def _heuristic(self, state: GameState) -> float:
        if check_win(state): return 0
        if not state.players or not any("-is-you" in r for r in state.rules): return 9999
        if any("-is-win" in r for r in state.rules):
            if state.winnables:
                player_pos = (state.players[0].x, state.players[0].y)
                win_positions = {(w.x, w.y) for w in state.winnables}
                return self._get_push_aware_distance(state, player_pos, win_positions)
            return 5000
        return 7500

    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        open_set = [HeapQEntry(self._heuristic(initial_state), 0, next(self.counter), [], initial_state)]; closed_set = {self._get_state_hash(initial_state): 0}
        nodes_expanded = 0
        for _ in range(iterations):
            if not open_set: break
            entry = heapq.heappop(open_set)
            nodes_expanded += 1
            if check_win(entry.state):
                print(f"  - Soluzione trovata dopo {nodes_expanded} espansioni.")
                return entry.actions
            if len(entry.actions) >= self.max_path_length: continue
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(direction, entry.state.copy()); h = self._get_state_hash(next_state)
                if h and entry.g_score + 1 >= closed_set.get(h, float('inf')): continue
                heuristic_score = self._heuristic(next_state)
                if heuristic_score >= 9999: continue
                closed_set[h] = entry.g_score + 1; heapq.heappush(open_set, HeapQEntry(entry.g_score + 1 + heuristic_score, entry.g_score + 1, next(self.counter), entry.actions + [direction], next_state))
        print(f"  - Ricerca fallita dopo {nodes_expanded}/{iterations} espansioni.")
        return None

# ... (Le classi _InternalRuleMakerAgent e DynamicPlannerAgent sono invariate, ma le riporto per completezza)
class _InternalRuleMakerAgent(_BaseAStarAgent):
    def __init__(self, target_rule: str, essential_rules: List[str]):
        super().__init__(); self.target_rule, self.essential_rules = target_rule, essential_rules
    def _heuristic(self, state: GameState) -> float:
        if any(rule not in state.rules for rule in self.essential_rules): return float('inf')
        if not state.players: return float('inf')
        if self.target_rule in state.rules: return 0
        try:
            noun, _, prop = self.target_rule.split('-'); noun_words, is_words, prop_words = find_word_objects(state, noun), find_word_objects(state, 'is'), find_word_objects(state, prop)
            if not all([noun_words, is_words, prop_words]): return float('inf')
            player_pos = (state.players[0].x, state.players[0].y)
            return min(manhattan_distance(player_pos, (n.x, n.y)) + manhattan_distance((n.x, n.y), (i.x, i.y)) - 1 + manhattan_distance((i.x, i.y), (p.x, p.y)) - 1 for n in noun_words for i in is_words for p in prop_words)
        except ValueError: return float('inf')
    def search(self, initial_state: GameState, iterations: int) -> Optional[List[Direction]]:
        open_set = [HeapQEntry(self._heuristic(initial_state), 0, next(self.counter), [], initial_state)]; closed_set = {self._get_state_hash(initial_state): 0}
        nodes_expanded = 0
        for _ in range(iterations):
            if not open_set: break
            entry = heapq.heappop(open_set)
            nodes_expanded += 1
            if self.target_rule in entry.state.rules: 
                print(f"  - Sottoproblema risolto dopo {nodes_expanded} espansioni.")
                return entry.actions
            if len(entry.actions) >= self.max_path_length: continue
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(direction, entry.state.copy()); h = self._get_state_hash(next_state)
                if h and entry.g_score + 1 >= closed_set.get(h, float('inf')): continue
                heuristic_score = self._heuristic(next_state)
                if heuristic_score == float('inf'): continue
                closed_set[h] = entry.g_score + 1; heapq.heappush(open_set, HeapQEntry(entry.g_score + 1 + heuristic_score, entry.g_score + 1, next(self.counter), entry.actions + [direction], next_state))
        print(f"  - Ricerca sottoproblema fallita dopo {nodes_expanded}/{iterations} espansioni.")
        return None

class RULE_MAKERAgent:
    def __init__(self):
        self.base_agent = _BaseAStarAgent()
    def _find_blockers_on_path(self, state: GameState) -> set:
        if not state.players or not state.winnables: return set()
        q = deque([(state.players[0].x, state.players[0].y)]); visited = {(state.players[0].x, state.players[0].y)}
        w, h = len(state.obj_map[0]), len(state.obj_map)
        path_found = False
        while q:
            x, y = q.popleft()
            if any(win_obj.x == x and win_obj.y == y for win_obj in state.winnables): path_found = True; break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    if state.obj_map[ny][nx] == ' ': q.append((nx, ny))
        if path_found: return set()
        blockers = set()
        for x, y in visited:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    obj = state.obj_map[ny][nx]
                    if isinstance(obj, GameObj): blockers.add(obj.name)
        return blockers
    def _deduce_subgoals(self, state: GameState) -> List[str]:
        print("🤔 Deducing subgoals with robust spatial analysis...")
        all_unpushable_phys_names = {obj.name for obj in state.phys if obj not in state.pushables}
        if not all_unpushable_phys_names:
             print("  - No unpushable objects found. No subgoals needed."); return []
        print(f"  - Analyzing potential problem objects: {all_unpushable_phys_names}")
        candidate_subgoals = []
        for obj_name in all_unpushable_phys_names:
            subgoal = f"{obj_name}-is-push"
            if all([find_word_objects(state, obj_name), find_word_objects(state, 'is'), find_word_objects(state, 'push')]):
                rule_maker = _InternalRuleMakerAgent(subgoal, []); cost = rule_maker._heuristic(state)
                if cost != float('inf'): candidate_subgoals.append({'subgoal': subgoal, 'cost': cost})
        if not candidate_subgoals:
            print("  - Found unpushable objects, but no words available to form PUSH rules."); return []
        best = min(candidate_subgoals, key=lambda x: x['cost'])
        print(f"  ✅ Best plan deduced: '{best['subgoal']}' with estimated cost {best['cost']:.2f}")
        return [best['subgoal']]
    def _solve_with_subgoals(self, initial_state: GameState, subgoals: List[str], iterations: int) -> Optional[List[Direction]]:
        full_solution, current_state = [], initial_state.copy()
        essential_rules = [r for r in initial_state.rules if "-is-you" in r or "-is-win" in r]
        print(f"Essential rules to preserve: {essential_rules}")
        for i, subgoal_rule in enumerate(subgoals):
            print(f"\n[Subgoal {i+1}/{len(subgoals)}] Creating rule: '{subgoal_rule}'")
            rule_maker = _InternalRuleMakerAgent(target_rule=subgoal_rule, essential_rules=essential_rules)
            subgoal_solution = rule_maker.search(current_state, iterations=iterations)
            if subgoal_solution is None: print(f"❌ FAILED to create rule '{subgoal_rule}'."); return None
            full_solution.extend(subgoal_solution); current_state = apply_solution(current_state, subgoal_solution)
            essential_rules.append(subgoal_rule)
        print("\n--- Subgoals met. Starting final search. ---")
        main_solution = self.base_agent.search(current_state, iterations=iterations)
        if main_solution is None: print("❌ FAILED to find final solution from intermediate state."); return None
        full_solution.extend(main_solution); print(f"✅ SUCCESS! Full solution found with {len(full_solution)} moves.")
        return full_solution
    def search(self, initial_state: GameState, iterations: int = 10000, **kwargs) -> Optional[List[Direction]]:
        print("--- Phase 1: Quick Probe ---"); quick_search_iterations = min(iterations, 2000)
        solution = self.base_agent.search(initial_state, iterations=quick_search_iterations)
        if solution is not None:
            print("✅ Quick probe successful. Level is simple."); return solution
        print("\n--- Phase 2: Quick Probe Failed. Analyzing for complex strategy. ---")
        subgoals = self._deduce_subgoals(initial_state)
        if not subgoals:
            print("Could not deduce a valid plan. The level might be too complex or requires a different strategy.")
            print("--- Last resort: Running full-depth standard A* search. ---")
            return self.base_agent.search(initial_state, iterations=iterations)
        print(f"\n--- Phase 3: Plan execution with subgoals: {subgoals} ---")
        return self._solve_with_subgoals(initial_state, subgoals, iterations)