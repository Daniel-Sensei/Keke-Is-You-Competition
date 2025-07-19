# agents/robust_astar_AGENT.py
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
import heapq
import time

class ROBUST_ASTARAgent(BaseAgent):
    """Hybrid A* plus IDA* fallback with dynamic beam width for challenging levels."""
    def __init__(self):
        # Base A* settings
        self.possible_actions = [d for d in Direction if d != Direction.Undefined]
        self.max_time = 120.0
        self.initial_beam = 500
        self.beam_growth = 2  # double on each retry
        self.max_retries = 3
        self.max_path_length = 150

    def _get_state_hash(self, state: GameState) -> tuple:
        # reuse robust hash similar to enhanced version
        try:
            players = tuple(sorted((p.x, p.y) for p in state.players))
            objects = tuple(sorted((o.name, o.x, o.y) for o in state.phys if hasattr(o, 'name')))
            rules = tuple(sorted(state.rules))
            return (players, objects, rules)
        except:
            return tuple(str(o)[:20] for o in state.phys[:5])

    def _calculate_heuristic(self, state: GameState) -> float:
        # use manhattan to nearest winnable with penalty for deadlocks
        if check_win(state): return 0.0
        if not state.players or not state.winnables:
            return 50.0  # encourage exploration
        px, py = state.players[0].x, state.players[0].y
        min_dist = min(abs(px - w.x) + abs(py - w.y) for w in state.winnables)
        # penalty for loops: count repeated patterns in phys
        return min_dist

    def search(self, initial_state: GameState, iterations: int = None):
        start = time.time()
        beam = self.initial_beam
        # try multiple beam widths
        for attempt in range(self.max_retries):
            path = self._beam_astar(initial_state, beam, start)
            if path and self._validate_solution(initial_state, path):
                return path
            beam *= self.beam_growth
        # Final fallback to IDA*
        return self._ida_star(initial_state, start)

    def _beam_astar(self, initial: GameState, beam_width: int, start_time: float):
        # beam search variant of A*
        open_set = [(self._calculate_heuristic(initial), 0.0, self._get_state_hash(initial), initial, [])]
        closed = set()
        while open_set and time.time() - start_time < self.max_time:
            # prune beam
            open_set = sorted(open_set, key=lambda x: x[0])[:beam_width]
            f, g, hkey, state, path = heapq.heappop(open_set)
            if hkey in closed: continue
            closed.add(hkey)
            if check_win(state): return path
            for action in self.possible_actions:
                ns = advance_game_state(action, state.copy())
                nh = self._calculate_heuristic(ns)
                nk = self._get_state_hash(ns)
                if nk in closed or len(path) + 1 >= self.max_path_length: continue
                heapq.heappush(open_set, (g + 1 + nh, g + 1, nk, ns, path + [action]))
        return []

    def _ida_star(self, initial: GameState, start_time: float):
        # simple IDA* loop
        bound = self._calculate_heuristic(initial)
        path = []
        while True:
            t = self._dfs_ida(initial, 0, bound, path, start_time)
            if isinstance(t, list): return t
            if t == float('inf') or time.time() - start_time >= self.max_time:
                break
            bound = t
        return []

    def _dfs_ida(self, state, g, bound, path, start_time):
        f = g + self._calculate_heuristic(state)
        if f > bound: return f
        if check_win(state): return path.copy()
        if time.time() - start_time >= self.max_time: return float('inf')
        min_cost = float('inf')
        for action in self.possible_actions:
            path.append(action)
            ns = advance_game_state(action, state.copy())
            t = self._dfs_ida(ns, g + 1, bound, path, start_time)
            if isinstance(t, list): return t
            if t < min_cost: min_cost = t
            path.pop()
        return min_cost

    def _validate_solution(self, initial_state: GameState, solution):
        try:
            s = initial_state
            for act in solution:
                s = advance_game_state(act, s.copy())
                if check_win(s): return True
        except:
            return False
        return False
