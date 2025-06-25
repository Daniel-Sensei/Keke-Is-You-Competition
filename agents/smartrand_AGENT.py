"""
Random agent implementation for KekeAI.
This agent selects actions randomly and serves as a baseline for comparison with other algorithms.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List
import random
from tqdm import trange



class SMARTRANDAgent(BaseAgent):
    """
    A slightly smarter random agent that avoids obviously bad moves.
    This variant tries to avoid moving into walls or repeating the same position too frequently.
    """

    def __init__(self, seed: int = None, memory_size: int = 5):
        """
        Initialize the smart random agent.
        
        :param seed: Random seed for reproducible results (optional)
        :param memory_size: Number of recent states to remember to avoid cycles
        """
        if seed is not None:
            random.seed(seed)
        self.memory_size = memory_size

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Performs smart random search with basic cycle avoidance.
        
        :param initial_state: The initial game state
        :param iterations: Maximum number of actions to try
        :return: List of actions that lead to a solution (if found)
        """
        possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        
        current_state = initial_state.copy()
        actions_taken = []
        recent_states = []  # Keep track of recent states to avoid cycles
        
        for i in trange(iterations, desc="Smart random search"):
            # Check if we have won the game
            if check_win(current_state):
                return actions_taken
            
            # Get current state representation for cycle detection
            state_str = str(current_state)
            
            # Try to find a good random action (avoid recently visited states)
            attempts = 0
            max_attempts = len(possible_actions) * 2
            
            while attempts < max_attempts:
                random_action = random.choice(possible_actions)
                test_state = advance_game_state(random_action, current_state.copy())
                test_state_str = str(test_state)
                
                # If this state hasn't been visited recently, use this action
                if test_state_str not in recent_states:
                    current_state = test_state
                    actions_taken.append(random_action)
                    
                    # Update recent states memory
                    recent_states.append(state_str)
                    if len(recent_states) > self.memory_size:
                        recent_states.pop(0)
                    
                    break
                
                attempts += 1
            
            # If we couldn't find a good move, just take any random action
            if attempts >= max_attempts:
                random_action = random.choice(possible_actions)
                current_state = advance_game_state(random_action, current_state.copy())
                actions_taken.append(random_action)
                
                # Clear some memory to allow more exploration
                if len(recent_states) > self.memory_size // 2:
                    recent_states = recent_states[self.memory_size // 2:]
        
        # If no solution found within iteration limit
        return []