"""
Agente A* per Baba Is You.
Utilizza l'algoritmo A* con euristica ottimizzata e BFS come backup.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Set, Optional
import heapq
import time
from collections import deque


class AStarNode:
    """Nodo per l'algoritmo A*."""
    
    def __init__(self, state: GameState, path: List[Direction], g_cost: float, h_cost: float):
        self.state = state
        self.path = path
        self.g_cost = g_cost  # costo dal nodo iniziale
        self.h_cost = h_cost  # euristica
        self.f_cost = g_cost + h_cost  # costo totale
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
class KEVIN_ASTARAgent(BaseAgent):
    def __init__(self):
        self.visited: Set[str] = set()
        self.max_path_length = 40   # lunghezza massima del percorso limitata a 100
        self.max_time = 320.0       # tempo massimo in secondi

    def search(self, initial_state: GameState, iterations: int = 100) -> List[Direction]:
        #max_path_length dinamico in base alla dimensione della mappa
        # Mappa 6x6 → max_path_length ≈ 26
        # Mappa 10x10 → ≈ 42
        # Mappa 15x15 → ≈ 66
        width = len(initial_state.obj_map[0])
        height = len(initial_state.obj_map)
        self.max_path_length = int((width + height) * 1.8)

        astar_result = self._astar_search(initial_state, iterations*150)
        if astar_result:
            print(f"A* ha trovato soluzione in {len(astar_result)} mosse")
            return astar_result
        
        return []

    def _astar_search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """Esegue una ricerca A* per trovare la soluzione ottimale con limite di tempo e iterazioni."""
        self.visited.clear()
        
        # Inizializza il timer
        start_time = time.time()
        
        # Inizializza la coda di priorità con il nodo iniziale
        initial_h = self._heuristic(initial_state)
        initial_node = AStarNode(initial_state, [], 0.0, initial_h)
        
        open_list = [initial_node]
        heapq.heapify(open_list)
        
        iteration = 0
        
        #while open_list and iteration < iterations:
        while open_list:        
            # Controlla il limite di tempo
            if time.time() - start_time > self.max_time:
                print(f"A* interrotto per timeout dopo {self.max_time} secondi")
                break
            
            # Prendi il nodo con f_cost minore
            current_node = heapq.heappop(open_list)
            
            # Controlla se abbiamo vinto
            if check_win(current_node.state):
                elapsed_time = time.time() - start_time
                print(f"A* completato in {elapsed_time:.2f} secondi")
                return current_node.path
            
            # Controlla se abbiamo già visitato questo stato
            state_str = str(current_node.state)
            if state_str in self.visited:
                continue
            
            self.visited.add(state_str)
            
            # Se il percorso è troppo lungo, salta
            if len(current_node.path) >= self.max_path_length:
                continue
            
            # Espandi i nodi figli
            for move in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                # Simula la mossa
                next_state = advance_game_state(move, current_node.state.copy())
                
                # Calcola i costi
                g_cost = current_node.g_cost + 1.0  # ogni mossa costa 1
                h_cost = self._heuristic(next_state)
                
                # Crea il nuovo nodo
                new_path = current_node.path + [move]
                new_node = AStarNode(next_state, new_path, g_cost, h_cost)
                
                # Aggiungi alla coda se non visitato
                next_state_str = str(next_state)
                if next_state_str not in self.visited:
                    heapq.heappush(open_list, new_node)
        
        # A* fallito
        elapsed_time = time.time() - start_time
        print(f"A* terminato senza soluzione dopo {elapsed_time:.2f} secondi, iterazioni: {iteration}")
        return []

    def _manhattan_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calcola la distanza di Manhattan tra due punti."""
        return abs(x1 - x2) + abs(y1 - y2)

    def _smart_distance(self, player_x: int, player_y: int, target_x: int, target_y: int, state: GameState, max_search: int = 15) -> float:
        """
        Calcola la distanza Manhattan con penalità rapida per ostacoli sul percorso diretto.
        Molto più veloce del precedente BFS.
        """
        if player_x == target_x and player_y == target_y:
            return 0.0
        
        # Calcola distanza Manhattan base
        manhattan_dist = self._manhattan_distance(player_x, player_y, target_x, target_y)
        
        # Penalità rapida per ostacoli sul percorso diretto
        obstacle_penalty = 0.0
        
        # Controlla ostacoli sul percorso orizzontale diretto
        if player_y == target_y:
            x_start, x_end = (player_x + 1, target_x) if player_x < target_x else (target_x + 1, player_x)
            for x in range(x_start, x_end):
                if not self._is_navigable(x, player_y, state):
                    obstacle_penalty += 2.0
        
        # Controlla ostacoli sul percorso verticale diretto
        elif player_x == target_x:
            y_start, y_end = (player_y + 1, target_y) if player_y < target_y else (target_y + 1, player_y)
            for y in range(y_start, y_end):
                if not self._is_navigable(player_x, y, state):
                    obstacle_penalty += 2.0
        
        # Per percorsi diagonali, controlla solo alcuni punti chiave
        else:
            # Controlla punto intermedio orizzontale
            mid_x = target_x
            mid_y = player_y
            if not self._is_navigable(mid_x, mid_y, state):
                obstacle_penalty += 1.5
            
            # Controlla punto intermedio verticale
            mid_x = player_x
            mid_y = target_y
            if not self._is_navigable(mid_x, mid_y, state):
                obstacle_penalty += 1.5
        
        return manhattan_dist + obstacle_penalty

    def _is_navigable(self, x: int, y: int, state: GameState) -> bool:
        """Determina se una cella è navigabile dal player."""
        # Controlli bounds
        if (x < 0 or y < 0 or 
            y >= len(state.obj_map) or x >= len(state.obj_map[0])):
            return False
        
        # Controlla background per border
        if state.back_map[y][x] == '_':
            return False
        
        # Controlla oggetti che bloccano il movimento
        obj = state.obj_map[y][x]
        if obj != ' ':
            # Gli oggetti STOP bloccano sempre
            if hasattr(state, 'stoppables') and state.stoppables:
                for stop_obj in state.stoppables:
                    if stop_obj.x == x and stop_obj.y == y:
                        return False
            
            # Gli oggetti PUSH sono navigabili se possono essere spinti
            if hasattr(state, 'pushables') and state.pushables:
                for push_obj in state.pushables:
                    if push_obj.x == x and push_obj.y == y:
                        return True
            
            # Altri oggetti fisici sono considerati ostacoli
            if hasattr(obj, 'object_type') and obj.object_type.name == 'Physical':
                return False
        
        return True

    def _calculate_proximity_penalty(self, players: List, dangerous_objects: List, penalty_base: float) -> float:
        """Calcola penalità basata sulla vicinanza tra player e oggetti pericolosi."""
        if not players or not dangerous_objects:
            return 0.0
        
        total_penalty = 0.0
        for player in players:
            min_distance = float('inf')
            for dangerous_obj in dangerous_objects:
                distance = self._manhattan_distance(player.x, player.y, dangerous_obj.x, dangerous_obj.y)
                min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                # Penalità inversamente proporzionale alla distanza
                # Distanza 1 = penalità massima, distanza 5+ = penalità minima
                proximity_factor = max(0.2, 1.0 / max(1, min_distance))
                total_penalty += penalty_base * proximity_factor
        
        return total_penalty

    def _analyze_rule_interactions(self, state: GameState) -> float:
        """Analizza le interazioni tra regole per identificare situazioni problematiche."""
        penalty = 0.0
        rules = [rule.lower() for rule in state.rules]
        
        # Combinazioni pericolose
        dangerous_combinations = [
            (['move'], ['kill', 'hot', 'melt', 'sink']),  # MOVE + dangerous
            (['open'], ['shut']),  # OPEN + SHUT può causare stalli
            (['move'], ['stop']),  # MOVE + STOP può causare conflitti
        ]
        
        for primary_rules, secondary_rules in dangerous_combinations:
            has_primary = any(any(pr in rule for pr in primary_rules) for rule in rules)
            has_secondary = any(any(sr in rule for sr in secondary_rules) for rule in rules)
            
            if has_primary and has_secondary:
                # Penalità maggiore se ci sono più regole MOVE
                move_count = sum(1 for rule in rules if 'move' in rule)
                penalty += 15.0 + (move_count * 5.0)
        
        # Analisi specifica per regole MOVE
        move_objects = []
        for rule in state.rules:
            if 'move' in rule.lower():
                obj_type = rule.split('-')[0].lower()
                if obj_type in state.sort_phys:
                    move_objects.extend(state.sort_phys[obj_type])
        
        if move_objects:
            # Penalità se oggetti MOVE sono vicini a player o winnables
            targets = state.players + state.winnables
            for move_obj in move_objects:
                for target in targets:
                    distance = self._manhattan_distance(move_obj.x, move_obj.y, target.x, target.y)
                    if distance <= 3:  # Vicinanza pericolosa
                        penalty += 10.0 * (4 - distance)  # Più vicino = più pericoloso
        
        return min(penalty, 50.0)  # Limita la penalità massima

    def _heuristic(self, state: GameState) -> float:
        """Euristica migliorata per Baba Is You."""
        h = 0.0

        if not state.players:
            return 500.0

        # 1. DISTANZA INTELLIGENTE A WINNABLES (ora più veloce)
        if state.winnables:
            min_smart_dist = float('inf')
            for p in state.players:
                for w in state.winnables:
                    # Usa smart distance semplificata invece di BFS
                    dist = self._smart_distance(p.x, p.y, w.x, w.y, state)
                    min_smart_dist = min(min_smart_dist, dist)
            h += min_smart_dist * 2.0
        else:
            h += 300.0

        # 2. Analisi regole YOU (invariato)
        you_rules = sum(1 for r in state.rules if "you" in r.lower())
        if you_rules == 0:
            h += 200.0
        elif you_rules > 1:
            h -= 10.0

        # 3. PENALITÀ DINAMICHE PER REGOLE PERICOLOSE
        # Oggetti pericolosi con penalità dinamiche basate sulla vicinanza
        if state.killers:
            h += self._calculate_proximity_penalty(state.players, state.killers, 25.0)
        
        if state.sinkers:
            h += self._calculate_proximity_penalty(state.players, state.sinkers, 20.0)
        
        # Analisi regole HOT/MELT dinamiche
        hot_objects = []
        melt_objects = []
        for rule in state.rules:
            rule_lower = rule.lower()
            if 'hot' in rule_lower:
                obj_type = rule.split('-')[0]
                if obj_type in state.sort_phys:
                    hot_objects.extend(state.sort_phys[obj_type])
            elif 'melt' in rule_lower:
                obj_type = rule.split('-')[0]
                if obj_type in state.sort_phys:
                    melt_objects.extend(state.sort_phys[obj_type])
        
        if hot_objects and melt_objects:
            # Penalità se oggetti HOT e MELT sono vicini tra loro
            for hot_obj in hot_objects:
                for melt_obj in melt_objects:
                    distance = self._manhattan_distance(hot_obj.x, hot_obj.y, melt_obj.x, melt_obj.y)
                    if distance <= 3:
                        h += 15.0 * (4 - distance)

        # 4. Bonus per regole PUSH (invariato)
        push_rules = sum(1 for r in state.rules if "push" in r.lower())
        h -= push_rules * 2.0
        
        # 5. ANALISI AVANZATA DELLE INTERAZIONI TRA REGOLE
        h += self._analyze_rule_interactions(state)

        # 6. ANALISI PERCORSO E OSTACOLI MIGLIORATA (semplificata)
        if state.players and state.winnables:
            path_penalty = 0
            
            for p in state.players:
                for w in state.winnables:
                    # Usa smart distance semplificata per una valutazione più veloce
                    smart_dist = self._smart_distance(p.x, p.y, w.x, w.y, state)
                    manhattan_dist = self._manhattan_distance(p.x, p.y, w.x, w.y)
                    
                    # Se smart distance è maggiore di Manhattan, c'è complessità del percorso
                    if smart_dist > manhattan_dist:
                        path_penalty += (smart_dist - manhattan_dist) * 1.5
                    
                    # Bonus se il player ha oggetti PUSH vicini per superare ostacoli
                    if state.pushables:
                        nearby_pushables = sum(1 for push_obj in state.pushables 
                                            if self._manhattan_distance(p.x, p.y, push_obj.x, push_obj.y) <= 2)
                        path_penalty -= min(nearby_pushables * 1.5, 5.0)  # Max 5 punti di bonus
            
            h += min(path_penalty, 20.0)  # Aumenta il limite massimo

        return max(h, 0.1)