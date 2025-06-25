"""
Esempio di agente che utilizza BFS (Breadth-First Search) per risolvere i livelli.
Questo è un esempio di come strutturare un agente concreto.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List
from collections import deque
from tqdm import trange


class BFSAgent(BaseAgent):
    """
    Breadth-First Search implementation.
    """

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        queue = deque([(initial_state, [])])  # (current state, action history)
        visited = set()

        for i in trange(iterations):
            if not queue:
                break
            current_state, actions = queue.popleft()

            # Check if we have won the game
            if check_win(current_state):
                return actions

            # Mark this state as visited
            state_str = str(current_state)  # Serialize the state to check for duplicates
            if state_str in visited:
                continue
            visited.add(state_str)

            # Get all possible actions and apply them
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                next_state = advance_game_state(action, current_state.copy())
                if str(next_state) not in visited:
                    queue.append((next_state, actions + [action]))

        return []

    