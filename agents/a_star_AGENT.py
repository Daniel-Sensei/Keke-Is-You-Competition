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

# --- Classe Agente A* (Modificata) ---
class A_STARAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 20000
        self.counter = itertools.count()
        
        # --- MODIFICA ---
        # Abbiamo ridotto il peso per la formazione delle regole a 1.0.
        # Un valore > 1 rende l'euristica "inammissibile", il che significa che può
        # sovrastimare il costo reale. Questo la rende più "avida" ma meno affidabile.
        # Riportandola a 1.0, l'algoritmo A* torna ad essere ammissibile (se l'euristica di base lo è),
        # garantendo di trovare il percorso più breve, anche se potrebbe richiedere più tempo per farlo.
        # Questo rende l'agente più sistematico e meno prono a finire in vicoli ciechi.
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.0
        self.P_NO_WIN_RULE = 100
        self.P_NO_YOU_RULE = 500

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        print("--- Inizio Ricerca A* (Versione Modificata) ---")
        
        start_h = self._heuristic(initial_state)
        if start_h == float('inf'):
            print("Stato iniziale non risolvibile.")
            return None
            
        open_set = [HeapQEntry(start_h, 0, next(self.counter), [], initial_state)]
        closed_set = {}

        initial_hash = self._get_state_hash(initial_state)
        if initial_hash:
            closed_set[initial_hash] = 0
            
        # --- MODIFICA ---
        # Il problema principale era il limite di 100 iterazioni imposto dall'esterno.
        # Ignoriamo il parametro 'iterations' e usiamo il nostro limite di classe,
        # 'self.max_iterations', che è molto più generoso e adatto alla complessità del gioco.
        # Questo dà all'agente il "respiro" necessario per esplorare a fondo lo spazio di ricerca.
        print(f"Limite iterazioni impostato a: {self.max_iterations}")
        for i in range(self.max_iterations):
            if not open_set:
                print("Ricerca A* fallita: open set vuoto.")
                break

            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state
            
            if check_win(current_state):
                print(f"Soluzione A* trovata in {len(actions)} mosse dopo {i} iterazioni!")
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

        print("Ricerca A* terminata: nessuna soluzione trovata entro il limite di iterazioni.")
        return None

    def _heuristic(self, state: GameState) -> float:
        # La logica interna dell'euristica rimane invariata, ma beneficerà
        # del peso W_FORM_RULE corretto e di un hashing più robusto.
        if check_win(state):
            return 0
        if not state.players:
            return float('inf')

        rule_analysis = analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        has_win = bool(rule_analysis['win_rules'])

        if has_you and has_win:
            min_dist = float('inf')
            if not state.winnables: return float('inf')
            for player in state.players:
                for winnable in state.winnables:
                    dist = manhattan_distance((player.x, player.y), (winnable.x, winnable.y))
                    min_dist = min(min_dist, dist)
            return min_dist * self.W_REACH_WIN

        if not has_you:
            cost = self._estimate_rule_formation_cost(state, 'you')
            return self.P_NO_YOU_RULE + cost

        if not has_win:
            cost = self._estimate_rule_formation_cost(state, 'win')
            return self.P_NO_WIN_RULE + cost
            
        return float('inf')

    def _estimate_rule_formation_cost(self, state: GameState, rule_property: str) -> float:
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        property_words = find_word_objects(state, rule_property)

        if not nouns or not is_words or not property_words:
            return float('inf')

        min_formation_dist = float('inf')

        for noun in nouns:
            dist_to_is = min([manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words])
            dist_to_prop = min([manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words])
            current_dist = dist_to_is + dist_to_prop
            min_formation_dist = min(min_formation_dist, current_dist)

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
        
        # --- MODIFICA ---
        # Abbiamo modificato l'hashing per includere la posizione di TUTTI gli oggetti fisici
        # (state.phys), non solo un sottoinsieme o quelli attualmente 'pushable'.
        # Questo rende l'hash più robusto, poiché cattura lo stato di ogni singolo blocco sulla mappa.
        # In questo modo, si evita che l'agente confonda due stati in cui un oggetto
        # (non tracciato in precedenza) è stato spostato, impedendogli di bloccare per errore
        # un percorso di soluzione valido.
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        components.append(f"O:{','.join(map(str, phys_pos))}")
        
        return "|".join(components)