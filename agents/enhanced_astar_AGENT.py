# agents/enhanced_astar_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq
import time


class ENHANCED_ASTARAgent(BaseAgent):
    def __init__(self):
        self.possible_actions = [d for d in Direction if d != Direction.Undefined]
        self.max_time = 90.0  # Aumentato a 90 secondi
        self.max_path_length = 120  # Aumentato a 120 mosse
        self.beam_width = 8000  # Limita la ricerca ai migliori stati

    def _get_state_hash(self, game_state: GameState) -> tuple:
        """Hash ottimizzato per identificare stati unici."""
        try:
            # Hash più flessibile che considera anche le regole
            players = tuple(sorted((p.x, p.y) for p in game_state.players if hasattr(p, 'x') and hasattr(p, 'y')))
            
            # Considera anche la posizione degli oggetti fisici principali
            key_objects = []
            for obj in game_state.phys:
                if hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'name'):
                    # Considera solo alcuni tipi di oggetti per ridurre la complessità
                    if obj.name in ['flag_obj', 'baba_obj', 'rock_obj', 'skull_obj', 'wall_obj']:
                        key_objects.append((obj.name, obj.x, obj.y))
            
            key_objects = tuple(sorted(key_objects))
            rules = tuple(sorted(game_state.rules))
            
            return (players, key_objects, rules)
        except:
            # Fallback più robusto
            return tuple(sorted(str(obj)[:20] for obj in game_state.phys[:5]))

    def _is_dangerous_state(self, state: GameState) -> bool:
        """Rileva stati pericolosi dove il giocatore può morire."""
        if not state.players:
            return False
        
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
        return False

    def _calculate_heuristic(self, state: GameState) -> float:
        """Euristica adattiva che considera diverse condizioni di vittoria."""
        try:
            if check_win(state):
                return 0.0
            
            # Controllo stati pericolosi
            if self._is_dangerous_state(state):
                return 999.0
            
            # Se non abbiamo giocatori, penalità alta
            if not state.players:
                return 100.0
            
            # Euristica adattiva basata sul tipo di livello
            if state.winnables:
                # Livello classico con oggetti vincenti
                return self._heuristic_with_winnables(state)
            else:
                # Livello senza winnables iniziali - usa euristica esplorativa
                return self._heuristic_exploratory(state)
                
        except:
            return 20.0

    def _heuristic_with_winnables(self, state: GameState) -> float:
        """Euristica per livelli con winnables, migliorata con penalità per ostacoli e distanze elevate."""
        min_distance = float('inf')
        penalty = 0.0
        players_to_check = state.players[:3]
        winnables_to_check = state.winnables[:3]
        for player in players_to_check:
            if hasattr(player, 'x') and hasattr(player, 'y'):
                for winnable in winnables_to_check:
                    if hasattr(winnable, 'x') and hasattr(winnable, 'y'):
                        distance = abs(player.x - winnable.x) + abs(player.y - winnable.y)
                        # Penalità per oggetti pericolosi tra player e winnable
                        if hasattr(state, 'phys'):
                            for obj in state.phys:
                                if obj is player or obj is winnable:
                                    continue
                                if (player.x == winnable.x == obj.x and min(player.y, winnable.y) < obj.y < max(player.y, winnable.y)) or \
                                   (player.y == winnable.y == obj.y and min(player.x, winnable.x) < obj.x < max(player.x, winnable.x)):
                                    if hasattr(obj, 'name'):
                                        if 'kill' in obj.name or 'sink' in obj.name:
                                            penalty += 2.0
                                        if 'stop' in obj.name or 'push' in obj.name:
                                            penalty += 1.0
                        # Penalità leggera per distanze elevate
                        if distance > 6:
                            penalty += 0.5 * (distance - 6)
                        min_distance = min(min_distance, distance)
        return (min_distance if min_distance != float('inf') else 30.0) + penalty

    def _heuristic_exploratory(self, state: GameState) -> float:
        """Euristica esplorativa per livelli senza winnables iniziali."""
        # Incoraggia la formazione di nuove regole e la diversità degli stati
        base_score = 10.0
        
        # Bonus per avere più regole attive (potrebbero creare winnables)
        rule_bonus = -len(state.rules) * 0.5
        
        # Bonus per avere diversi tipi di oggetti
        object_types = set()
        for obj in state.phys:
            if hasattr(obj, 'name'):
                object_types.add(obj.name)
        diversity_bonus = -len(object_types) * 0.2
        
        # Penalità per avere troppi giocatori (potrebbero morire)
        player_penalty = max(0, len(state.players) - 5) * 2.0
        
        # Bonus per la posizione centrale (incoraggia l'esplorazione)
        center_bonus = 0.0
        if state.players:
            avg_x = sum(p.x for p in state.players if hasattr(p, 'x')) / len(state.players)
            avg_y = sum(p.y for p in state.players if hasattr(p, 'y')) / len(state.players)
            # Assumiamo una mappa 10x10 (centrata su 5,5)
            center_distance = abs(avg_x - 5) + abs(avg_y - 5)
            center_bonus = center_distance * 0.1
        
        final_score = base_score + rule_bonus + diversity_bonus + player_penalty + center_bonus
        return max(1.0, final_score)

    def _fallback_dfs(self, initial_state: GameState, max_depth: int = 120) -> List[Direction]:
        """DFS di fallback ottimizzato con euristica."""
        from collections import deque
        stack = deque()
        visited = set()
        start_time = time.time()
        
        stack.append((initial_state, []))
        print("[DEBUG] Avvio DFS fallback con euristica...")
        
        visited_limit = 30000  # Aumentato per livelli difficili
        best_heuristic = float('inf')
        
        while stack and (time.time() - start_time) < 30.0 and len(visited) < visited_limit:
            state, path = stack.pop()
            
            if len(path) > max_depth:
                continue
                
            h = self._get_state_hash(state)
            if h in visited:
                continue
            visited.add(h)
            
            # Traccia il progresso
            current_h = self._calculate_heuristic(state)
            if current_h < best_heuristic:
                best_heuristic = current_h
                if len(visited) % 1000 == 0:
                    print(f"[DEBUG] DFS: Miglior euristica {best_heuristic:.1f} a profondità {len(path)}")
            
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
            for h_val, action, next_state in reversed(actions_with_heuristic[:4]):  # Solo le 4 migliori
                new_path = path + [action]
                stack.append((next_state, new_path))
        
        print(f"[DEBUG] DFS fallback completato. Visitati {len(visited)} stati, best_h: {best_heuristic:.1f}")
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

    def _prune_open_set(self, open_set: List) -> List:
        """Mantiene solo i migliori stati per limitare la memoria."""
        if len(open_set) <= self.beam_width:
            return open_set
        
        # Ordina per f-score e mantiene solo i migliori
        open_set.sort(key=lambda x: x[0])
        return open_set[:self.beam_width]

    def search(self, initial_state: GameState, iterations: int = None) -> List[Direction]:
        """Ricerca A* semplificata con iterazioni illimitate ma percorso limitato."""
        print("[DEBUG] Avvio ricerca A* semplificata...")
        start_time = time.time()
        
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}
        
        init_hash = self._get_state_hash(initial_state)
        
        print(f"[DEBUG] Configurazione: iterazioni illimitate, percorso max {self.max_path_length}, timeout {self.max_time}s")
        
        g_costs[init_hash] = 0.0
        h = self._calculate_heuristic(initial_state)
        heapq.heappush(open_set, (h, 0.0, init_hash, initial_state, []))
        
        iteration_count = 0
        
        while open_set and (time.time() - start_time) < self.max_time:
            f, g, hash_key, state, path = heapq.heappop(open_set)
            
            # Salta se percorso troppo lungo
            if len(path) >= self.max_path_length:
                continue
                
            if hash_key in closed_set: 
                continue
            closed_set.add(hash_key)
            
            if check_win(state):
                elapsed = time.time() - start_time
                print(f"[DEBUG] Soluzione trovata in {len(path)} mosse, {iteration_count} iterazioni, {elapsed:.2f}s")
                return path
            
            # Esplora azioni
            for action in self.possible_actions:
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
            
            # Potatura dell'open set per limitare l'uso della memoria
            open_set = self._prune_open_set(open_set)
            
            iteration_count += 1
            
            # Progress report ogni 1000 iterazioni
            if iteration_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"[DEBUG] Iterazione {iteration_count}, elapsed {elapsed:.1f}s, open_set: {len(open_set)}")
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] A* terminato: {iteration_count} iterazioni, {elapsed:.2f}s")
        
        # Fallback a DFS se A* non trova soluzione
        print("[DEBUG] A* fallito, tentativo con DFS...")
        return self._fallback_dfs(initial_state, self.max_path_length)
