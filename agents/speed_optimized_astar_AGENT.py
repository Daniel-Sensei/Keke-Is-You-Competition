import heapq
import itertools
import time
from typing import List, Dict, Tuple, Set, Optional

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win, name_to_character)

# --- Versione super-ottimizzata per velocità ---

def fast_manhattan(x1, y1, x2, y2):
    """Manhattan distance inline ottimizzata."""
    return abs(x1 - x2) + abs(y1 - y2)

class FastEntry:
    """Entry compatta per heapq."""
    __slots__ = ['f', 'g', 'actions', 'state']
    
    def __init__(self, f_score, g_score, actions, state):
        self.f = f_score
        self.g = g_score
        self.actions = actions
        self.state = state

    def __lt__(self, other):
        return self.f < other.f

class SPEED_OPTIMIZED_ASTARAgent(BaseAgent):
    """Agente A* ottimizzato per velocità pura."""
    
    def __init__(self):
        super().__init__()
        self.max_iterations = 100000
        
        # Pesi ottimizzati empiricamente
        self.win_weight = 1.0
        self.rule_weight = 3.0
        self.no_win_penalty = 50
        self.no_you_penalty = 200

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        """Search ottimizzato con early termination aggressiva."""
        start_time = time.time()
        
        # Early win check
        if check_win(initial_state):
            return []
        
        # Heuristic iniziale
        start_h = self._quick_heuristic(initial_state)
        if start_h > 500:  # Cutoff aggressivo
            return None

        open_set = [FastEntry(start_h, 0, [], initial_state)]
        visited = set()
        
        iterations_done = 0
        best_f = start_h

        for _ in range(self.max_iterations):
            # Check tempo ogni 5000 iterazioni
            iterations_done += 1
            if iterations_done % 5000 == 0:
                if time.time() - start_time > 180:  # 3 minuti max
                    return None

            if not open_set:
                break

            current = heapq.heappop(open_set)
            
            # Early pruning se f-score troppo alto
            if current.f > best_f + 100:
                continue
                
            if current.f < best_f:
                best_f = current.f

            # Win check
            if check_win(current.state):
                return current.actions

            # Hash semplificato per evitare cicli
            state_key = self._simple_hash(current.state)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Limita profondità
            if current.g > 50:
                continue

            # Genera successori
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(direction, current.state.copy())
                next_g = current.g + 1
                
                # Skip stati impossibili
                if not next_state.players:
                    continue
                
                h_score = self._quick_heuristic(next_state)
                
                # Pruning aggressivo
                if h_score > 300:
                    continue
                
                f_score = next_g + h_score
                
                # Solo stati promettenti
                if f_score < best_f + 50:
                    new_actions = current.actions + [direction]
                    heapq.heappush(open_set, FastEntry(f_score, next_g, new_actions, next_state))

        return None

    def _quick_heuristic(self, state: GameState) -> float:
        """Euristica ultra-veloce."""
        if check_win(state):
            return 0
        
        if not state.players:
            return 999
        
        # Check regole veloce
        has_you = any('-is-you' in rule for rule in state.rules)
        has_win = any('-is-win' in rule for rule in state.rules)
        
        if has_you and has_win:
            if state.winnables:
                # Distanza minima ai win objects
                player = state.players[0]
                min_dist = min(fast_manhattan(player.x, player.y, w.x, w.y) for w in state.winnables)
                return min_dist * self.win_weight
            else:
                return self.no_win_penalty
        
        if not has_you:
            return self.no_you_penalty
            
        if not has_win:
            return self.no_win_penalty
            
        return 999

    def _simple_hash(self, state: GameState) -> str:
        """Hash minimalista per controllo cicli."""
        if not state.players:
            return "dead"
        
        # Solo posizione primo giocatore + conteggio oggetti
        player = state.players[0]
        return f"{player.x},{player.y},{len(state.words)},{len(state.phys)}"
