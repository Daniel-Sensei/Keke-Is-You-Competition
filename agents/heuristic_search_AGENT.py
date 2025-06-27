from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Dict, Tuple
import heapq
from tqdm import trange

#
# VERSIONE 5.0 - APPROCCIO SEMPLICE E ROBUSTO
#
# I tentativi precedenti di replicare l'agente complesso del paper senza l'ottimizzazione
# dei pesi sono falliti. Questa versione adotta la strategia dell'agente "Default"
# menzionato nel paper: un A* con una funzione euristica molto più semplice e fondamentale,
# basata solo su tre distanze chiave[cite: 236, 237].
# L'obiettivo è la robustezza invece della complessità.
#
class HEURISTIC_SEARCHAgent(BaseAgent):
    """
    Agente A* con una funzione euristica semplice e robusta, focalizzata
    sulle distanze dagli oggetti di interesse primario.
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Pesi per le tre euristiche di distanza fondamentali.
        """
        if weights is None:
            self.weights = {
                'DistToWin': 10.0,
                'DistToWords': 5.0,
                'DistToPushables': 2.0,
            }
        else:
            self.weights = weights

    def _get_avg_min_dist(self, player_positions: List[Tuple[int, int]], target_positions: List[Tuple[int, int]]) -> float:
        """
        Calcola la distanza media minima dai giocatori ai target.
        Se non ci sono target, la distanza è 0 (non c'è nulla da raggiungere).
        Se non ci sono giocatori, la distanza è infinita.
        """
        if not player_positions:
            return float('inf')
        if not target_positions:
            return 0.0

        total_min_dist = 0
        for p_pos in player_positions:
            # Calcola la distanza di Manhattan dal giocatore al target più vicino
            min_dist = min(abs(p_pos[0] - t_pos[0]) + abs(p_pos[1] - t_pos[1]) for t_pos in target_positions)
            total_min_dist += min_dist
            
        return total_min_dist / len(player_positions)

    def _calculate_heuristic_cost(self, state: GameState) -> float:
        """
        Calcola il costo euristico h(n). Un valore BASSO indica uno stato promettente.
        La funzione è la somma pesata delle tre distanze fondamentali.
        """
        # Se si vince, il costo è zero (o negativo per incentivare)
        if check_win(state):
            return -1.0
        
        player_positions = [(p.x, p.y) for p in state.players]
        
        # 1. Distanza dagli oggetti che fanno vincere
        winnable_positions = [(g.x, g.y) for g in state.winnables]
        dist_win = self._get_avg_min_dist(player_positions, winnable_positions)

        # 2. Distanza dalle parole (per formare regole)
        word_positions = [(w.x, w.y) for w in (state.words + state.is_connectors)]
        dist_words = self._get_avg_min_dist(player_positions, word_positions)

        # 3. Distanza dagli oggetti spingibili
        pushable_positions = [(p.x, p.y) for p in state.pushables]
        dist_pushables = self._get_avg_min_dist(player_positions, pushable_positions)
        
        # Calcola il costo totale come somma pesata
        cost = (self.weights['DistToWin'] * dist_win +
                self.weights['DistToWords'] * dist_words +
                self.weights['DistToPushables'] * dist_pushables)
                
        return cost

    def search(self, initial_state: GameState, iterations: int = 2000) -> List[Direction]:
        """
        Esegue una ricerca A*. L'obiettivo è minimizzare f(n) = g(n) + h(n).
        """
        g_score = 0
        h_score = self._calculate_heuristic_cost(initial_state)
        f_score = g_score + h_score
        
        # Coda di priorità: (f_score, g_score, azioni, stato)
        priority_queue = [(f_score, g_score, [], initial_state)]
        
        # Dizionario degli stati visitati: {state_str: g_score}
        visited = {str(initial_state): g_score}

        for _ in trange(iterations, desc="Heuristic Search (Robust V5)"):
            if not priority_queue:
                break

            _, g, actions, current_state = heapq.heappop(priority_queue)

            if check_win(current_state):
                return actions

            # Esplora le 5 azioni possibili
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                # Se un'azione non cambia lo stato (es. sbattere contro un muro), Wait è un'opzione migliore
                if action != Direction.Wait and str(advance_game_state(action, current_state.copy())) == str(current_state):
                    continue

                next_state = advance_game_state(action, current_state.copy())
                next_state_str = str(next_state)
                new_g_score = g + 1

                # Se abbiamo già trovato un percorso più breve per questo stato, lo ignoriamo
                if visited.get(next_state_str, float('inf')) <= new_g_score:
                    continue
                
                # Se il nuovo stato porta a una sconfitta, lo ignoriamo
                if not next_state.players:
                    continue
                
                # Calcola il nuovo punteggio f(n) e aggiungi alla coda
                new_h_score = self._calculate_heuristic_cost(next_state)
                new_f_score = new_g_score + new_h_score
                
                visited[next_state_str] = new_g_score
                heapq.heappush(priority_queue, (new_f_score, new_g_score, actions + [action], next_state))

        return []