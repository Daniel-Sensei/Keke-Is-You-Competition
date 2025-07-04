import heapq
import itertools
from typing import List, Dict, Tuple, Set, Optional

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win, name_to_character)

# --- Funzioni Ausiliarie (invariate) ---

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calcola la distanza di Manhattan tra due punti."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_word_objects(state: GameState, word_name: str) -> List[GameObj]:
    """Trova tutti gli oggetti parola con un dato nome."""
    return [w for w in state.words if w.name == word_name]

def analyze_current_rules(state: GameState) -> Dict[str, List[str]]:
    """Analizza le regole correnti e le categorizza."""
    rule_analysis = {'you_rules': [], 'win_rules': [], 'other_rules': []}
    for rule in state.rules:
        if '-is-you' in rule:
            rule_analysis['you_rules'].append(rule)
        elif '-is-win' in rule:
            rule_analysis['win_rules'].append(rule)
        else:
            rule_analysis['other_rules'].append(rule)
    return rule_analysis

# --- Classe per HeapQ Entry (invariata) ---
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

# --- Classe Agente A* (Corretta) ---
class A_STARAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 200000
        self.counter = itertools.count()
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.0
        self.P_NO_WIN_RULE = 100
        self.P_NO_YOU_RULE = 500

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        start_h = self._heuristic(initial_state)
        if start_h == float('inf'):
           return None
            
        open_set = [HeapQEntry(start_h, 0, next(self.counter), [], initial_state)]
        closed_set = {}

        initial_hash = self._get_state_hash(initial_state)
        if initial_hash:
            closed_set[initial_hash] = 0
            
        for i in range(self.max_iterations):
            if not open_set:
                break

            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state
            
            if check_win(current_state):
                return actions

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
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
            # Caso 1: Ci sono oggetti "winnable" sulla mappa.
            # L'obiettivo è raggiungerli.
            if state.winnables:
                min_dist = float('inf')
                for player in state.players:
                    for winnable in state.winnables:
                        dist = manhattan_distance((player.x, player.y), (winnable.x, winnable.y))
                        min_dist = min(min_dist, dist)
                return min_dist * self.W_REACH_WIN
            # --- MODIFICA CHIAVE ---
            # Caso 2: Esiste una regola '... IS WIN', ma nessun oggetto corrispondente.
            # (es. 'FLAG IS WIN' ma non ci sono FLAG).
            # L'obiettivo diventa creare l'oggetto mancante.
            else:
                # Estrae il nome dell'oggetto da creare (es. 'flag' da 'flag-is-win').
                winnable_noun = rule_analysis['win_rules'][0].split('-is-')[0]
                # Calcola il costo per formare una regola come 'ROCK IS FLAG'.
                cost = self._estimate_rule_formation_cost(state, winnable_noun)
                if cost == float('inf'):
                    return float('inf')
                # Aggiunge una penalità perché creare una regola è più complesso che muoversi.
                return self.P_NO_WIN_RULE + cost

        # Se non c'è una regola '... IS YOU', calcola il costo per crearla.
        if not has_you:
            cost = self._estimate_rule_formation_cost(state, 'you')
            return self.P_NO_YOU_RULE + cost

        # Se non c'è una regola '... IS WIN', calcola il costo per crearla.
        if not has_win:
            cost = self._estimate_rule_formation_cost(state, 'win')
            return self.P_NO_WIN_RULE + cost
            
        return float('inf')

    # --- MODIFICA CHIAVE ---
    # La funzione è stata generalizzata per stimare il costo di formare una regola
    # con qualsiasi proprietà, non solo 'you' o 'win'.
    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        """Stima il costo per creare una regola del tipo '[NOME] IS [target_property_name]'."""
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        # Trova le parole corrispondenti alla proprietà obiettivo (es. 'win', 'you', 'flag').
        property_words = find_word_objects(state, target_property_name)

        if not nouns or not is_words or not property_words:
            return float('inf')

        min_formation_dist = float('inf')

        # Calcola la distanza minima per allineare NOME + IS + PROPRIETÀ.
        for noun in nouns:
            # Non consideriamo regole auto-referenziali inutili come 'FLAG IS FLAG'.
            if noun.name == target_property_name:
                continue
            dist_to_is = min([manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words])
            dist_to_prop = min([manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words])
            current_dist = dist_to_is + dist_to_prop
            min_formation_dist = min(min_formation_dist, current_dist)

        # Aggiunge la distanza del giocatore più vicino necessario per spingere i blocchi.
        if state.players:
            player_dist = min([manhattan_distance((p.x, p.y), (n.x, n.y)) for p in state.players for n in nouns])
            return (min_formation_dist + player_dist) * self.W_FORM_RULE
        
        return min_formation_dist * self.W_FORM_RULE

    def _get_state_hash(self, state: GameState) -> Optional[str]:
        if not state.players:
            return "NO_PLAYERS_STATE"
            
        components = []
        
        player_pos = sorted([(p.x, p.y) for p in state.players])
        components.append(f"P:{','.join(map(str, player_pos))}")

        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        components.append(f"W:{','.join(map(str, word_pos))}")
        
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        components.append(f"O:{','.join(map(str, phys_pos))}")
        
        return "|".join(components)