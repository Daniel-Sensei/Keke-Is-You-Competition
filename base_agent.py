"""
Modello di base per l'agente.
Fornisce una classe di base che gli agenti possono estendere.
"""
from baba import GameState, Direction
from abc import ABC, abstractmethod
from typing import List


class BaseAgent(ABC):
    """
    Classe base per gli agenti di KekeAI.
    Gli agenti concreti devono estendere questa classe e implementare il metodo solve.
    """
    
    @abstractmethod
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Method to perform search.
        :param initial_state: The initial state of the game.
        :param iterations: Maximum depth for algorithms like DFS.
        :return: List of actions that lead to a solution (if found).
        """
        pass