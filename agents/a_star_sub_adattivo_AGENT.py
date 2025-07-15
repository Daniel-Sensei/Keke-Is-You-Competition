import heapq
import itertools
import time
from typing import List, Dict, Tuple, Optional

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, advance_game_state, check_win)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def find_word_objects(state: GameState, word_name: str) -> List[GameObj]:
    return [w for w in state.words if w.name == word_name]


def analyze_current_rules(state: GameState) -> Dict[str, List[str]]:
    rule_analysis = {'you_rules': [], 'win_rules': [], 'other_rules': []}
    for rule in state.rules:
        if '-is-you' in rule:
            rule_analysis['you_rules'].append(rule)
        elif '-is-win' in rule:
            rule_analysis['win_rules'].append(rule)
        else:
            rule_analysis['other_rules'].append(rule)
    return rule_analysis


class HeapQEntry:
    def __init__(self, priority: float, g_score: int, tie_breaker: int, actions: List[Direction], state: GameState):
        self.priority = priority
        self.g_score = g_score
        self.tie_breaker = tie_breaker
        self.actions = actions
        self.state = state

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.g_score != other.g_score:
            return self.g_score > other.g_score
        return self.tie_breaker < other.tie_breaker


class A_STAR_SUB_ADATTIVOAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 200000
        self.counter = itertools.count()
        self.max_path_length = 40
        self.learning_rate = 0.0001

        self.weights = {
            'dist': 1.0,
            'win': 0.5,
            'you': 0.5
        }

        self.backup_heuristic_mode = False
        self.backup_trigger_time = 200

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        start_time = time.time()
        self.backup_heuristic_mode = False

        goal_stack = self._identify_goal_stack(initial_state)
        total_plan: List[Direction] = []
        current_state = initial_state.copy()

        for subgoal in goal_stack:
            print(f"🎯 Risolvendo sotto-obiettivo: {subgoal['name']}")
            plan = self._search_for_subgoal(current_state, subgoal, start_time)
            if not plan:
                print(f"❌ Fallito su obiettivo: {subgoal['name']}")
                print("🔁 Fallback: eseguo A* standard su stato iniziale...")
                return self._search_plain(initial_state, start_time)
            for move in plan:
                current_state = advance_game_state(move, current_state.copy())
            total_plan.extend(plan)

            if check_win(current_state):
                break

        return total_plan if check_win(current_state) else self._search_plain(initial_state, start_time)

    def _search_plain(self, initial_state: GameState, start_time: float) -> Optional[List[Direction]]:
        open_set = []
        closed_set = {}

        start_h, contribs = self._combined_heuristic(initial_state)
        if start_h == float('inf'):
            return None

        heapq.heappush(open_set, HeapQEntry(start_h, 0, next(self.counter), [], initial_state))
        closed_set[self._get_state_hash(initial_state)] = 0

        for _ in range(self.max_iterations):
            elapsed = time.time() - start_time
            if elapsed > 400:
                return None
            if not self.backup_heuristic_mode and elapsed > self.backup_trigger_time:
                print("🟡 Timeout parziale superato: attivo modalità euristica secondaria.")
                self.backup_heuristic_mode = True

            if not open_set:
                break

            current = heapq.heappop(open_set)
            g_score, actions, state = current.g_score, current.actions, current.state

            if check_win(state):
                self._update_heuristic_weights(True, contribs)
                return actions

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.max_path_length:
                    continue

                next_state = advance_game_state(direction, state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)

                if new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue

                h_score, contribs = self._combined_heuristic(next_state)
                if h_score == float('inf'):
                    continue

                closed_set[state_hash] = new_g_score
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))

        self._update_heuristic_weights(False, contribs)
        return None

    def _identify_goal_stack(self, state: GameState) -> List[Dict]:
        rule_analysis = analyze_current_rules(state)
        goals = []

        if not rule_analysis['you_rules']:
            goals.append({'name': 'CREA YOU', 'target_rule': 'you'})
        if not rule_analysis['win_rules']:
            goals.append({'name': 'CREA WIN', 'target_rule': 'win'})
        if rule_analysis['you_rules'] and rule_analysis['win_rules']:
            goals.append({'name': 'RAGGIUNGI OGGETTO VINCENTE', 'target_rule': None})

        return goals

    def _search_for_subgoal(self, start_state: GameState, subgoal: Dict, start_time: float) -> Optional[List[Direction]]:
        open_set = []
        closed_set = {}
        self.counter = itertools.count()

        if subgoal['target_rule'] == 'you':
            h_score = self.heuristic_rule_formation(start_state, 'you')
        elif subgoal['target_rule'] == 'win':
            h_score = self.heuristic_rule_formation(start_state, 'win')
        else:
            h_score = self.heuristic_distance_to_win(start_state)

        if h_score == float('inf'):
            return None

        heapq.heappush(open_set, HeapQEntry(h_score, 0, next(self.counter), [], start_state))
        closed_set[self._get_state_hash(start_state)] = 0

        while open_set and (time.time() - start_time) < 400:
            elapsed = time.time() - start_time
            if not self.backup_heuristic_mode and elapsed > self.backup_trigger_time:
                print("🟡 Timeout parziale superato: attivo modalità euristica secondaria.")
                self.backup_heuristic_mode = True

            current = heapq.heappop(open_set)
            g_score, actions, state = current.g_score, current.actions, current.state

            if self._goal_reached(state, subgoal):
                return actions

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.max_path_length:
                    continue

                next_state = advance_game_state(direction, state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)

                if new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue

                if subgoal['target_rule'] == 'you':
                    h_score = self.heuristic_rule_formation(next_state, 'you')
                elif subgoal['target_rule'] == 'win':
                    h_score = self.heuristic_rule_formation(next_state, 'win')
                else:
                    h_score = self.heuristic_distance_to_win(next_state)

                if h_score == float('inf'):
                    continue

                closed_set[state_hash] = new_g_score
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))

        return None

    def _goal_reached(self, state: GameState, subgoal: Dict) -> bool:
        rules = analyze_current_rules(state)
        if subgoal['target_rule'] == 'you':
            return bool(rules['you_rules'])
        elif subgoal['target_rule'] == 'win':
            return bool(rules['win_rules'])
        else:
            return check_win(state)

    def heuristic_distance_to_win(self, state: GameState) -> float:
        if not state.players or not state.winnables:
            return float('inf')
        return min(
            manhattan_distance((p.x, p.y), (w.x, w.y))
            for p in state.players for w in state.winnables
        )

    def heuristic_rule_formation(self, state: GameState, target: str) -> float:
        nouns = [w for w in state.words if w.name != 'is']
        is_words = find_word_objects(state, 'is')
        prop_words = find_word_objects(state, target)
        if not nouns or not is_words or not prop_words:
            return float('inf')

        best = float('inf')
        for noun in nouns:
            d1 = min([manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words])
            d2 = min([manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in prop_words])
            total = d1 + d2
            if state.players:
                total += min([manhattan_distance((p.x, p.y), (noun.x, noun.y)) for p in state.players])
            best = min(best, total)
        return best

    def _combined_heuristic(self, state: GameState) -> Tuple[float, Dict[str, float]]:
        if check_win(state):
            return 0, {'dist': 0, 'win': 0, 'you': 0}

        rule_analysis = analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        has_win = bool(rule_analysis['win_rules'])

        h_dist = self.heuristic_distance_to_win(state) if has_you and has_win else float('inf')
        h_win = self.heuristic_rule_formation(state, 'win') if not has_win else 0
        h_you = self.heuristic_rule_formation(state, 'you') if not has_you else 0

        contribs = {'dist': h_dist if h_dist != float('inf') else 0, 'win': h_win, 'you': h_you}

        if self.backup_heuristic_mode:
            total_score = h_win + h_you
        else:
            total_score = (
                self.weights['dist'] * contribs['dist'] +
                self.weights['win'] * contribs['win'] +
                self.weights['you'] * contribs['you']
            )

        return total_score, contribs

    def _update_heuristic_weights(self, solved: bool, contribs: Dict[str, float]):
        α = self.learning_rate
        for k in self.weights:
            delta = α * contribs[k] if solved else -α * contribs[k]
            self.weights[k] = max(0.01, self.weights[k] + delta)

        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

    def _get_state_hash(self, state: GameState) -> str:
        if not state.players:
            return "NO_PLAYERS"
        player_pos = sorted([(p.x, p.y) for p in state.players])
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        return f"P:{player_pos}|W:{word_pos}|O:{phys_pos}"
