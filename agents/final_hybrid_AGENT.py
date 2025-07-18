import heapq
import itertools
import time
from typing import List, Dict, Tuple, Optional, Callable

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state, check_win)

# --- Funzioni Ausiliarie e Classe per la Coda di Priorità ---
def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_word_objects(state: GameState, word_name: str) -> List[GameObj]:
    return [w for w in state.words if w.name == word_name]

class HeapQEntry:
    def __init__(self, priority: float, g_score: int, tie_breaker: int, actions: List[Direction], state: GameState):
        self.priority, self.g_score, self.tie_breaker, self.actions, self.state = \
            priority, g_score, tie_breaker, actions, state

    def __lt__(self, other):
        if self.priority != other.priority: return self.priority < other.priority
        if self.g_score != other.g_score: return self.g_score > other.g_score
        return self.tie_breaker < other.tie_breaker

# --- Agente Pianificatore Strategico con Analisi di Raggiungibilità ---

class FINAL_HYBRIDAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Parametri per la Fase 1: A* Veloce
        self.FAST_SEARCH_MAX_ITER = 30000
        
        # Parametri per la Fase 2: Pianificatore
        self.MAX_SUBGOALS = 15
        self.MAX_ITER_PER_SUBGOAL = 200000
        self.MAX_PATH_PER_SUBGOAL = 75

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        self.start_time = time.time()
        
        # --- FASE 1: Tentativo con A* Veloce e Diretto ---
        print("--- 🚀 Fase 1: Avvio ricerca veloce A*... ---")
        fast_solution = self._run_a_star_worker(initial_state, self.FAST_SEARCH_MAX_ITER, check_win, self._simple_heuristic, self.MAX_PATH_PER_SUBGOAL)
        
        if fast_solution is not None:
            print(f"✅ Soluzione trovata rapidamente in {len(fast_solution)} mosse!")
            return fast_solution

        # --- FASE 2: Attivazione del Pianificatore Strategico ---
        print("\n--- 🧠 Fase 2: A* diretto fallito. Avvio del Planner Strategico... ---")
        
        current_state = initial_state
        full_solution = []
        
        for i in range(self.MAX_SUBGOALS):
            print(f"\n--- 🗺️ Ciclo di Pianificazione #{i+1} ---")
            
            if check_win(current_state):
                print(f"✅ OBIETTIVO FINALE RAGGIUNTO! Lunghezza totale: {len(full_solution)}")
                return full_solution

            subgoal_func, subgoal_name = self._identify_strategic_subgoal(current_state)
            if subgoal_func is None:
                print("❌ Impossibile identificare un sotto-obiettivo valido. Pianificazione fallita.")
                return None
            
            print(f"🎯 Nuovo sotto-obiettivo strategico: {subgoal_name}")

            path_to_subgoal = self._run_a_star_worker(current_state, self.MAX_ITER_PER_SUBGOAL, subgoal_func, lambda s: 0, self.MAX_PATH_PER_SUBGOAL)

            if path_to_subgoal is None:
                print(f"❌ Impossibile risolvere il sotto-obiettivo: {subgoal_name}. L'agente si arrende.")
                return None
            
            print(f"✔️ Sotto-obiettivo '{subgoal_name}' raggiunto in {len(path_to_subgoal)} mosse.")
            full_solution.extend(path_to_subgoal)
            
            for move in path_to_subgoal:
                current_state = advance_game_state(move, current_state)
        
        print("⚠️ Raggiunto limite massimo di sotto-obiettivi.")
        return None

    def _flood_fill(self, state: GameState, start_pos: Tuple[int, int]) -> set:
        """Esegue un flood-fill per trovare tutte le caselle raggiungibili."""
        reachable = set()
        q = [start_pos]
        visited = {start_pos}
        
        stoppables = {(obj.x, obj.y) for obj in state.phys if obj.is_stopped}
        h, w = len(state.obj_map), len(state.obj_map[0])

        while q:
            x, y = q.pop(0)
            reachable.add((x,y))
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in visited or (nx, ny) in stoppables:
                    continue
                visited.add((nx,ny))
                q.append((nx, ny))
        return reachable

    def _identify_strategic_subgoal(self, state: GameState) -> Tuple[Optional[Callable[[GameState], bool]], str]:
        """Il cuore del pianificatore: analizza lo stato e sceglie l'obiettivo giusto."""
        if not state.players:
            return (lambda s: any('-is-you' in r for r in s.rules), "Creare una regola '... IS YOU'")

        player_pos = (state.players[0].x, state.players[0].y)
        reachable_tiles = self._flood_fill(state, player_pos)
        
        # Analisi degli ostacoli e degli strumenti
        is_wall_stop_active = any('wall-is-stop' in r for r in state.rules)
        is_win_rule_active = any('-is-win' in r for r in state.rules)
        
        # --- Logica per quando si è bloccati ---
        final_goal_reachable = any(w.name == 'flag' and (w.x, w.y) in reachable_tiles for w in state.phys) or \
                               any(w.name in ['wall', 'stop'] and (w.x, w.y) in reachable_tiles for w in state.words)

        if is_wall_stop_active and not final_goal_reachable:
            # SONO IN TRAPPOLA. Cerco strumenti per cambiare le regole.
            # In questo livello, lo strumento è formare "ROCK IS PUSH".
            is_push_words = find_word_objects(state, 'is') + find_word_objects(state, 'push')
            if len(is_push_words) == 2:
                is_word = is_push_words[0] if is_push_words[0].name == 'is' else is_push_words[1]
                # L'obiettivo è portare la parola ROCK sopra la parola IS.
                target_pos = (is_word.x, is_word.y - 1)
                
                def rock_is_push_formed(s: GameState):
                    rock_word = find_word_objects(s, 'rock')
                    return rock_word and (rock_word[0].x, rock_word[0].y) == target_pos

                return (rock_is_push_formed, "Formare la regola 'ROCK IS PUSH'")
        
        # --- Logica standard se non si è bloccati ---
        if is_wall_stop_active:
            return (lambda s: not any('wall-is-stop' in r for r in s.rules), "Rompere la regola 'WALL IS STOP'")
        
        if not is_win_rule_active:
            return (lambda s: any('-is-win' in r for r in s.rules), "Creare una regola '... IS WIN'")

        return (check_win, "Raggiungere l'oggetto WIN")

    def _simple_heuristic(self, state: GameState) -> float:
        if not state.players: return float('inf')
        if state.winnables:
            return min(manhattan_distance((p.x, p.y), (w.x, w.y)) for p in state.players for w in state.winnables)
        return 100

    def _run_a_star_worker(self, initial_state: GameState, max_iter: int, goal_func: Callable, heuristic_func: Callable, max_path: int) -> Optional[List[Direction]]:
        """A* generico usato sia per la ricerca veloce che per i sotto-obiettivi."""
        counter = itertools.count()
        h_start = heuristic_func(initial_state)
        if h_start == float('inf'): return None

        open_set = [HeapQEntry(h_start, 0, next(counter), [], initial_state)]
        closed_set = {self._get_state_hash(initial_state): 0}
        
        for _ in range(max_iter):
            if not open_set: break
            
            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state

            if goal_func(current_state): return actions
            if len(actions) >= max_path: continue

            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)

                if new_g_score >= closed_set.get(state_hash, float('inf')): continue
                
                h_score = heuristic_func(next_state)
                if h_score == float('inf'): continue

                closed_set[state_hash] = new_g_score
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(counter), actions + [direction], next_state))
        return None

    def _get_state_hash(self, state: GameState) -> str:
        phys_pos = sorted([(obj.name, obj.x, obj.y) for obj in state.phys])
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        return f"P:{phys_pos}|W:{word_pos}"