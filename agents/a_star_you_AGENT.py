import heapq
import itertools
import time
from typing import List, Dict, Tuple, Optional

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win, name_to_character)

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

class A_STAR_YOUAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 200000
        self.counter = itertools.count()
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.0
        self.P_NO_WIN_RULE = 100
        self.P_SUBOPTIMAL_YOU_RULE = 200
        self.max_path_length = 50

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        start_time = time.time()
        time_limit = 400

        start_h = self._heuristic(initial_state)
        if start_h == float('inf'):
            return None

        open_set = [HeapQEntry(start_h, 0, next(self.counter), [], initial_state)]
        closed_set = {}

        initial_hash = self._get_state_hash(initial_state)
        if initial_hash:
            closed_set[initial_hash] = 0

        for _ in range(self.max_iterations):
            if time.time() - start_time > time_limit:
                print("⏰ Tempo massimo raggiunto.")
                return 0

            if not open_set:
                break

            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state

            if check_win(current_state):
                return actions

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.max_path_length:
                    continue

                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)

                if state_hash and new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue

                h_score = self._heuristic(next_state)
                if h_score == float('inf'):
                    continue

                if state_hash:
                    closed_set[state_hash] = new_g_score

                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))

        print("⚠️ Nessuna soluzione trovata entro il limite di 60 mosse.")
        return None

    def _heuristic(self, state: GameState) -> float:
        if check_win(state):
            return 0
        if not state.players:
            return float('inf')

        rule_analysis = analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        has_win = bool(rule_analysis['win_rules'])

        if has_you and has_win:
            if state.winnables:
                min_dist = min(
                    manhattan_distance((p.x, p.y), (w.x, w.y))
                    for p in state.players for w in state.winnables
                )
                return min_dist * self.W_REACH_WIN

            win_noun = rule_analysis['win_rules'][0].split('-is-')[0]
            cost = self._estimate_rule_formation_cost(state, win_noun)
            return self.P_NO_WIN_RULE + cost if cost != float('inf') else float('inf')

        # Analizza se si può cambiare o creare una regola is-you promettente
        cost = float('inf')
        possible_you_targets = self._find_possible_you_targets(state)
        for target in possible_you_targets:
            c = self._estimate_rule_formation_cost(state, target)
            cost = min(cost, c)

        if cost < float('inf'):
            return self.P_SUBOPTIMAL_YOU_RULE + cost

        return float('inf')

    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        property_words = find_word_objects(state, target_property_name)

        if not nouns or not is_words or not property_words:
            return float('inf')

        min_formation_dist = float('inf')
        for noun in nouns:
            if noun.name == target_property_name:
                continue
            dist_to_is = min(manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words)
            dist_to_prop = min(manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words)
            total_dist = dist_to_is + dist_to_prop
            min_formation_dist = min(min_formation_dist, total_dist)

        if state.players:
            player_dist = min(manhattan_distance((p.x, p.y), (n.x, n.y)) for p in state.players for n in nouns)
            return (min_formation_dist + player_dist) * self.W_FORM_RULE

        return min_formation_dist * self.W_FORM_RULE

    def _find_possible_you_targets(self, state: GameState) -> List[str]:
        # Cerca parole che potrebbero diventare il soggetto di una regola is-you
        return list(set(w.name for w in state.words if w.name not in ['is', 'you']))

    def _get_state_hash(self, state: GameState) -> Optional[str]:
        if not state.players:
            return "NO_PLAYERS_STATE"

        components = []
        player_pos = sorted((p.x, p.y) for p in state.players)
        components.append(f"P:{','.join(map(str, player_pos))}")

        word_pos = sorted((w.name, w.x, w.y) for w in state.words)
        components.append(f"W:{','.join(map(str, word_pos))}")

        phys_pos = sorted((o.name, o.x, o.y) for o in state.phys)
        components.append(f"O:{','.join(map(str, phys_pos))}")

        return "|".join(components)
