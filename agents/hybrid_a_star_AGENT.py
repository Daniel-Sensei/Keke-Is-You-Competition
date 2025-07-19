import heapq
import itertools
import time
from typing import List, Dict, Tuple, Optional

# Import delle classi e funzioni necessarie dal simulatore "Baba Is You"
from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win)

# --- Funzioni Ausiliarie (Invariate) ---
def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_word_objects(state: GameState, word_name: str) -> List[GameObj]:
    return [w for w in state.words if w.name == word_name]

def analyze_current_rules(state: GameState) -> Dict[str, List[str]]:
    rule_analysis = {'you_rules': [], 'win_rules': [], 'other_rules': []}
    for rule in state.rules:
        if '-is-you' in rule: rule_analysis['you_rules'].append(rule)
        elif '-is-win' in rule: rule_analysis['win_rules'].append(rule)
        else: rule_analysis['other_rules'].append(rule)
    return rule_analysis

# --- Classe per la Coda di Priorità (HeapQ) ---
class HeapQEntry:
    def __init__(self, priority: float, g_score: int, tie_breaker: int, actions: List[Direction], state: GameState):
        self.priority, self.g_score, self.tie_breaker, self.actions, self.state = \
            priority, g_score, tie_breaker, actions, state

    def __lt__(self, other):
        if self.priority != other.priority: return self.priority < other.priority
        if self.g_score != other.g_score: return self.g_score > other.g_score
        return self.tie_breaker < other.tie_breaker

# --- Agente A* Definitivo con Dead-End Detection ---
class HYBRID_A_STARAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.counter = itertools.count()
        # Parametri di ricerca
        self.MAX_PATH_LENGTH = 100
        self.TIME_LIMIT_S = 300
        self.MAX_ITERATIONS = 500000
        # Pesi e penalità per l'euristica
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.2
        self.P_NO_WIN_RULE = 100
        self.P_NO_YOU_RULE = 500

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        self.start_time = time.time()
        print("--- 🚀 Avvio Ultimate A* Agent con Dead-End Detection ---")
        
        # Inizializza le mappe di raggiungibilità e gli oggetti bloccanti per lo stato iniziale
        self._update_blocking_objects(initial_state)

        h_start = self._heuristic(initial_state)
        if h_start == float('inf'):
            print("❌ Stato iniziale già identificato come vicolo cieco.")
            return None

        open_set = [HeapQEntry(h_start, 0, next(self.counter), [], initial_state)]
        closed_set = {self._get_state_hash(initial_state): 0}
        
        for _ in range(self.MAX_ITERATIONS):
            if time.time() - self.start_time > self.TIME_LIMIT_S:
                print("⏰ Tempo massimo raggiunto.")
                return None
            
            if not open_set: break

            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state

            if check_win(current_state):
                print(f"✅ Soluzione trovata in {len(actions)} mosse!")
                return actions

            if len(actions) >= self.MAX_PATH_LENGTH: continue

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                
                # Aggiorna le informazioni di blocco per il nuovo stato
                self._update_blocking_objects(next_state)

                state_hash = self._get_state_hash(next_state)
                if new_g_score >= closed_set.get(state_hash, float('inf')): continue
                
                h_score = self._heuristic(next_state)
                if h_score == float('inf'): continue

                closed_set[state_hash] = new_g_score
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))
        
        print("❌ Nessuna soluzione trovata nei limiti.")
        return None

    def _update_blocking_objects(self, state: GameState):
        """
        Pre-calcola quali oggetti sono bloccanti per le analisi di raggiungibilità.
        Questa funzione arricchisce l'oggetto 'state' con nuove informazioni.
        """
        state.stoppable_objects = set()
        for obj in state.phys:
            if obj.is_stopped:
                state.stoppable_objects.add((obj.x, obj.y))
        
        # Considera anche i bordi della mappa come bloccanti
        h, w = len(state.obj_map), len(state.obj_map[0])
        for y in range(h):
            state.stoppable_objects.add((-1, y))
            state.stoppable_objects.add((w, y))
        for x in range(w):
            state.stoppable_objects.add((x, -1))
            state.stoppable_objects.add((x, h))


    def _is_dead_end(self, state: GameState) -> bool:
        """
        Controlla se lo stato attuale è un vicolo cieco.
        Restituisce True se è impossibile vincere da questo stato.
        """
        # --- Check 1: Esistono le parole necessarie per vincere? ---
        rule_analysis = analyze_current_rules(state)
        has_you_rule = bool(rule_analysis['you_rules'])
        has_win_rule = bool(rule_analysis['win_rules'])

        # Se non ho una regola YOU e non posso crearla, è un vicolo cieco.
        if not has_you_rule and not find_word_objects(state, 'you'):
            return True
        # Se non ho una regola WIN e non posso crearla, è un vicolo cieco.
        if not has_win_rule and not find_word_objects(state, 'win'):
            return True

        # --- Check 2: Le parole chiave sono intrappolate? ---
        # Una parola è intrappolata se ha 4 vicini bloccanti (muri, oggetti STOP o altre parole)
        all_words = state.words
        blocking_positions = state.stoppable_objects.union((w.x, w.y) for w in all_words)

        for word in all_words:
            neighbors = [(word.x+1, word.y), (word.x-1, word.y), (word.x, word.y+1), (word.x, word.y-1)]
            if all(pos in blocking_positions for pos in neighbors):
                 # Se la parola intrappolata è l'unica del suo tipo, è un problema serio.
                 if len(find_word_objects(state, word.name)) == 1:
                    # Se questa parola è essenziale (es. l'unico "IS" o l'unico "WIN")
                    # allora è un vicolo cieco.
                    if word.name in ['is', 'you', 'win']:
                         return True
        
        # --- Check 3 (opzionale, più complesso): Il giocatore può raggiungere gli obiettivi? ---
        # Si potrebbe implementare un flood-fill per vedere se c'è un percorso
        # tra un oggetto YOU e un oggetto WIN o le parole per formare le regole.
        
        return False

    def _heuristic(self, state: GameState) -> float:
        """
        Euristica potenziata che prima controlla i vicoli ciechi.
        """
        if self._is_dead_end(state):
            return float('inf')
        
        if check_win(state):
            return 0
        
        # (La logica dell'euristica potenziata rimane simile a quella precedente)
        rule_analysis = analyze_current_rules(state)
        you_nouns = [rule.split('-is-')[0] for rule in rule_analysis['you_rules']]
        win_nouns = [rule.split('-is-')[0] for rule in rule_analysis['win_rules']]
        
        cost_you = 0
        if not you_nouns:
            cost_you = self._estimate_rule_formation_cost(state, 'you')
            if cost_you == float('inf'): return float('inf')

        cost_win = 0
        if not win_nouns:
            cost_win = self._estimate_rule_formation_cost(state, 'win')
            if cost_win == float('inf'): return float('inf')
        
        cost_reach = float('inf')
        player_objs = [p for name in you_nouns for p in state.sort_phys.get(name, [])] if you_nouns else state.players
        
        if player_objs and win_nouns:
            winnable_objs = [w for name in win_nouns for w in state.sort_phys.get(name, [])]
            if winnable_objs:
                min_dist = min(manhattan_distance((p.x, p.y), (w.x, w.y)) for p in player_objs for w in winnable_objs)
                cost_reach = min_dist
        
        if cost_reach == float('inf'):
            return (cost_you * self.W_FORM_RULE + cost_win * self.W_FORM_RULE + self.P_NO_WIN_RULE)

        return (cost_you * self.W_FORM_RULE + cost_win * self.W_FORM_RULE + cost_reach * self.W_REACH_WIN)


    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        """Stima la distanza di Manhattan per formare una regola 'NOUN-IS-PROPERTY'."""
        # NOTA: Per un'ulteriore evoluzione, questa funzione potrebbe usare un A*
        # per calcolare il costo *reale* di spostamento, tenendo conto degli ostacoli.
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        property_words = find_word_objects(state, target_property_name)

        if not nouns or not is_words or not property_words:
            return float('inf')

        min_formation_dist = float('inf')
        for n in nouns:
            if n.name == target_property_name: continue
            dist_n_is = min(manhattan_distance((n.x, n.y), (i.x, i.y)) for i in is_words)
            dist_is_prop = min(manhattan_distance((i.x, i.y), (p.x, p.y)) for i in is_words for p in property_words)
            min_formation_dist = min(min_formation_dist, dist_n_is + dist_is_prop)

        return min_formation_dist

    def _get_state_hash(self, state: GameState) -> str:
        phys_pos = sorted([(obj.name, obj.x, obj.y) for obj in state.phys])
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        return f"PHYS:{phys_pos}|WORDS:{word_pos}"