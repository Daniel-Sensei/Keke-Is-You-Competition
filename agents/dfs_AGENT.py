"""
Esempio di agente che utilizza DFS (Depth-First Search) per risolvere i livelli.
Questo è un esempio di come strutturare un agente concreto.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List

class DFSAgent(BaseAgent):
    """
    Un agente che utilizza la ricerca in profondità (DFS) per trovare una soluzione.
    """
    
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        stack = [(initial_state, [])]  # (current state, action history)
        visited = set()

        while stack:
            current_state, actions = stack.pop()

            # Check if we have won the game
            if check_win(current_state):
                return actions

            # Mark this state as visited
            state_str = str(current_state)  # Serialize the state to check for duplicates
            if state_str in visited:
                continue
            visited.add(state_str)

            # Get all possible actions and apply them
            if len(actions) < iterations:
                for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                    next_state = advance_game_state(action, current_state.copy())
                    if str(next_state) not in visited:
                        stack.append((next_state, actions + [action]))

        return []