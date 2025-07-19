# agents/enhanced_astar_v2_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq
import time


class ENHANCED_ASTAR_V2_Agent(BaseAgent):
    def __init__(self):
        self.possible_actions = [d for d in Direction if d != Direction.Undefined]
        self.max_time = 90.0  # Aumentato a 90 secondi
        self.max_path_length = 120  # Aumentato a 120 mosse
        self.beam_width = 10000  # Limita la ricerca ai migliori N stati

    def _get_state_hash(self, game_state: GameState) -> tuple:
        """Hash ottimizzato per identificare stati unici."""
        try:
            # Hash solo degli elementi più importanti
            players = tuple(sorted((p.x, p.y) for p in game_state.players if hasattr(p, 'x') and hasattr(p, 'y')))
            winnables = tuple(sorted((w.x, w.y) for w in game_state.winnables if hasattr(w, 'x') and hasattr(w, 'y')))
            
            # Per i pushables, considera solo la posizione se non sono troppi
            if len(game_state.pushables) <= 20:
                pushables = tuple(sorted((p.x, p.y) for p in game_state.pushables if hasattr(p, 'x') and hasattr(p, 'y')))
            else:
                pushables = ()
            
            # Rules come stringa concatenata per efficienza
            rules = ''.join(sorted(game_state.rules))
            
            return (players, winnables, pushables, rules)
        except:
            # Fallback più robusto
            return tuple(sorted(str(obj) for obj in game_state.phys[:10]))

    def _is_dangerous_state(self, state: GameState) -> bool:
        """Rileva stati pericolosi dove il giocatore può morire."""
        if not state.players:
            return False
        
        try:
            for player in state.players:
                if hasattr(player, 'x') and hasattr(player, 'y'):
                    # Controlla oggetti killer nella stessa posizione
                    for killer in state.killers:
                        if hasattr(killer, 'x') and hasattr(killer, 'y'):
                            if killer.x == player.x and killer.y == player.y:
                                return True
                    # Controlla oggetti sinker
                    for sinker in state.sinkers:
                        if hasattr(sinker, 'x') and hasattr(sinker, 'y'):
                            if sinker.x == player.x and sinker.y == player.y:
                                return True
        except:
            pass
        return False

    def _calculate_heuristic(self, state: GameState) -> float:
        """Euristica semplice ma ottimizzata."""
        try:
            if check_win(state):
                return 0.0
            
            # Controllo stati pericolosi
            if self._is_dangerous_state(state):
                return 999.0
            
            # Se non abbiamo giocatori, penalità alta
            if not state.players:
                return 100.0
            
            # Se non abbiamo oggetti vincenti, penalità media
            if not state.winnables:
                return 50.0
            
            # Calcolo distanza Manhattan minima ottimizzato
            min_distance = float('inf')
            
            # Limita il numero di calcoli per prestazioni
            players_to_check = state.players[:5]  # Massimo 5 giocatori
            winnables_to_check = state.winnables[:5]  # Massimo 5 winnables
            
            for player in players_to_check:
                if hasattr(player, 'x') and hasattr(player, 'y'):
                    for winnable in winnables_to_check:
                        if hasattr(winnable, 'x') and hasattr(winnable, 'y'):
                            distance = abs(player.x - winnable.x) + abs(player.y - winnable.y)
                            min_distance = min(min_distance, distance)
            
            return min_distance if min_distance != float('inf') else 30.0
        except:
            return 20.0

    def _prune_open_set(self, open_set: List) -> List:
        """Mantiene solo i migliori stati per limitare la memoria."""
        if len(open_set) <= self.beam_width:
            return open_set
        
        # Ordina per f-score e mantiene solo i migliori
        open_set.sort(key=lambda x: x[0])
        return open_set[:self.beam_width]

    def _fallback_dfs(self, initial_state: GameState, max_depth: int = 120) -> List[Direction]:
        """DFS di fallback ottimizzato."""
        from collections import deque
        stack = deque()
        visited = set()
        start_time = time.time()
        
        stack.append((initial_state, []))
        print("[DEBUG] Avvio DFS fallback ottimizzato...")
        
        visited_limit = 50000  # Limita gli stati visitati
        
        while stack and (time.time() - start_time) < 30.0 and len(visited) < visited_limit:
            state, path = stack.pop()
            
            if len(path) > max_depth:
                continue
                
            h = self._get_state_hash(state)
            if h in visited:
                continue
            visited.add(h)
            
            if check_win(state):
                print(f"[DEBUG] Soluzione trovata con DFS fallback in {len(path)} mosse!")
                return path
            
            # Esplora azioni con euristica per ordinamento
            actions_with_heuristic = []
            for action in self.possible_actions:
                try:
                    next_state = advance_game_state(action, state.copy())
                    h_val = self._calculate_heuristic(next_state)
                    if h_val < 999.0:  # Evita stati pericolosi
                        actions_with_heuristic.append((h_val, action, next_state))
                except:
                    continue
            
            # Ordina per euristica (migliori prima)
            actions_with_heuristic.sort(key=lambda x: x[0])
            
            # Aggiungi alla stack (in ordine inverso perché è LIFO)
            for h_val, action, next_state in reversed(actions_with_heuristic[:3]):  # Solo le 3 migliori
                new_path = path + [action]
                stack.append((next_state, new_path))
        
        print(f"[DEBUG] DFS fallback completato. Visitati {len(visited)} stati.")
        return []

    def _validate_solution(self, initial_state: GameState, solution: List[Direction]) -> bool:
        """Valida che una soluzione risolva effettivamente il puzzle."""
        if not solution:
            return False
        
        try:
            current_state = initial_state
            for i, action in enumerate(solution):
                current_state = advance_game_state(action, current_state.copy())
                if check_win(current_state):
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] Errore nella validazione: {e}")
            return False

    def _optimize_solution(self, initial_state: GameState, solution: List[Direction]) -> List[Direction]:
        """Ottimizza una soluzione rimuovendo mosse inutili."""
        if not solution or len(solution) <= 1:
            return solution
        
        try:
            current_state = initial_state
            optimized = []
            
            for i, action in enumerate(solution):
                new_state = advance_game_state(action, current_state.copy())
                
                # Se raggiungiamo la vittoria, tronca qui
                if check_win(new_state):
                    optimized.append(action)
                    print(f"[DEBUG] Soluzione ottimizzata da {len(solution)} a {len(optimized)} mosse")
                    return optimized
                
                optimized.append(action)
                current_state = new_state
            
            return optimized
        except Exception as e:
            print(f"[ERROR] Errore nell'ottimizzazione: {e}")
            return solution

    def search(self, initial_state: GameState, iterations: int = None) -> List[Direction]:
        """Ricerca A* ottimizzata con beam search."""
        print("[DEBUG] Avvio ricerca A* ottimizzata v2...")
        start_time = time.time()
        
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}
        
        init_hash = self._get_state_hash(initial_state)
        
        print(f"[DEBUG] Config: max_path={self.max_path_length}, timeout={self.max_time}s, beam_width={self.beam_width}")
        
        g_costs[init_hash] = 0.0
        h = self._calculate_heuristic(initial_state)
        heapq.heappush(open_set, (h, 0.0, init_hash, initial_state, []))
        
        iteration_count = 0
        best_heuristic = float('inf')
        
        while open_set and (time.time() - start_time) < self.max_time:
            # Pruning della open_set se troppo grande
            if len(open_set) > self.beam_width:
                open_set = self._prune_open_set(open_set)
                heapq.heapify(open_set)
            
            f, g, hash_key, state, path = heapq.heappop(open_set)
            
            # Salta se percorso troppo lungo
            if len(path) >= self.max_path_length:
                continue
                
            if hash_key in closed_set: 
                continue
            closed_set.add(hash_key)
            
            # Traccia progresso
            current_h = self._calculate_heuristic(state)
            if current_h < best_heuristic:
                best_heuristic = current_h
                print(f"[DEBUG] Miglior euristica: {best_heuristic:.1f} a profondità {len(path)}")
            
            if check_win(state):
                elapsed = time.time() - start_time
                print(f"[DEBUG] Soluzione trovata in {len(path)} mosse, {iteration_count} iterazioni, {elapsed:.2f}s")
                return self._optimize_solution(initial_state, path)
            
            # Esplora azioni
            for action in self.possible_actions:
                try:
                    new_state = advance_game_state(action, state.copy())
                    new_hash = self._get_state_hash(new_state)
                    new_g = g + 1.0
                    
                    # Skip se già abbiamo un percorso migliore
                    if new_g >= g_costs.get(new_hash, float('inf')):
                        continue
                    
                    # Skip se percorso troppo lungo
                    if len(path) + 1 >= self.max_path_length:
                        continue
                    
                    g_costs[new_hash] = new_g
                    h = self._calculate_heuristic(new_state)
                    
                    # Skip stati pericolosi
                    if h >= 999.0:
                        continue
                    
                    new_path = path + [action]
                    f_score = new_g + h
                    heapq.heappush(open_set, (f_score, new_g, new_hash, new_state, new_path))
                except:
                    continue
            
            iteration_count += 1
            
            # Progress report più frequente
            if iteration_count % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"[DEBUG] Iter {iteration_count}, elapsed {elapsed:.1f}s, open_set: {len(open_set)}, best_h: {best_heuristic:.1f}")
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] A* terminato: {iteration_count} iterazioni, {elapsed:.2f}s")
        
        # Fallback a DFS migliorato
        print("[DEBUG] A* fallito, tentativo con DFS ottimizzato...")
        return self._fallback_dfs(initial_state, self.max_path_length)
