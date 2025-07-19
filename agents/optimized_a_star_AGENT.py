import heapq
import itertools
from typing import List, Optional, Dict, Tuple

from agents.a_star_AGENT import A_STARAgent
from baba import GameState, Direction, advance_game_state, check_win


class OPTIMIZED_A_STARAgent(A_STARAgent):
    """
    Optimized version of A_STARAgent that caches heuristic computations
    and uses parent pointers for efficient path reconstruction.
    """
    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        # Initialize heuristic and hash caches
        start_hash = self._get_state_hash(initial_state)
        start_h = self._heuristic(initial_state)
        if start_h == float('inf'):
            return None

        open_heap = []  # heap elements: (f_score, g_score, tie_breaker, state_hash, state)
        counter = itertools.count()
        heapq.heappush(open_heap, (start_h, 0, next(counter), start_hash, initial_state))

        came_from: Dict[str, Tuple[str, Direction]] = {}
        g_scores: Dict[str, int] = {start_hash: 0}
        heuristic_cache: Dict[str, float] = {start_hash: start_h}

        for _ in range(self.max_iterations):
            if not open_heap:
                break

            f_score, g_score, _, current_hash, current_state = heapq.heappop(open_heap)

            if check_win(current_state):
                # Reconstruct path of actions
                path: List[Direction] = []
                h = current_hash
                while h in came_from:
                    prev_hash, action = came_from[h]
                    path.append(action)
                    h = prev_hash
                return list(reversed(path))

            # Explore neighbors
            for direction in (Direction.Up, Direction.Down, Direction.Left, Direction.Right):
                next_state = advance_game_state(direction, current_state.copy())
                next_hash = self._get_state_hash(next_state)
                new_g = g_score + 1

                if next_hash in g_scores and new_g >= g_scores[next_hash]:
                    continue

                # Heuristic lookup or compute
                h = heuristic_cache.get(next_hash)
                if h is None:
                    h = self._heuristic(next_state)
                    heuristic_cache[next_hash] = h
                if h == float('inf'):
                    continue

                g_scores[next_hash] = new_g
                came_from[next_hash] = (current_hash, direction)
                f = new_g + h
                heapq.heappush(open_heap, (f, new_g, next(counter), next_hash, next_state))

        return None
