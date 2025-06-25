"""
Random agent implementation for KekeAI.
This agent selects actions randomly and serves as a baseline for comparison with other algorithms.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List
import random
from tqdm import trange


class RANDAgent(BaseAgent):
    """
    Random action selection agent.
    This agent randomly selects actions until it finds a solution or reaches the iteration limit.
    Useful as a baseline for comparing the performance of more sophisticated algorithms.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the random agent.
        
        :param seed: Random seed for reproducible results (optional)
        """
        if seed is not None:
            random.seed(seed)

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Performs random search by selecting actions randomly until a solution is found.
        
        :param initial_state: The initial game state
        :param iterations: Maximum number of random actions to try
        :return: List of actions that lead to a solution (if found)
        """
        # Available actions
        possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        
        current_state = initial_state.copy()
        actions_taken = []
        
        for i in trange(iterations, desc="Random search"):
            # Check if we have won the game
            if check_win(current_state):
                return actions_taken
            
            # Select a random action
            random_action = random.choice(possible_actions)
            
            # Apply the action and get the new state
            new_state = advance_game_state(random_action, current_state.copy())
            
            # Update current state and action history
            current_state = new_state
            actions_taken.append(random_action)
            
            # Optional: Add some basic loop detection to avoid getting stuck
            # If we've taken too many actions, we might be in a loop
            if len(actions_taken) > iterations * 0.8:  # 80% of max iterations
                # Try to reset occasionally by making a different choice
                if i % 10 == 0:  # Every 10 steps, try a different strategy
                    # Prefer movement actions over wait
                    movement_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
                    random_action = random.choice(movement_actions)
                    new_state = advance_game_state(random_action, current_state.copy())
                    current_state = new_state
                    actions_taken.append(random_action)
        
        # If no solution found within iteration limit
        return []
