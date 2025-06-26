import math
import random
from typing import List, Optional

"""
Monte Carlo Tree Search (MCTS) Agent for KekeAI.

This agent implements the MCTS algorithm to find solutions for Baba Is You style puzzles.
The MCTS algorithm consists of four main phases:
1. Selection: Traverse the tree from the root using a tree policy (UCB1) to find a promising node.
2. Expansion: If the selected node is not terminal and not fully expanded, create one or more child nodes.
3. Simulation (Rollout): From a newly expanded node (or the selected node), simulate a random playout 
   until a terminal state is reached or a depth limit is hit.
4. Backpropagation: Update the visit counts and win scores for all nodes from the simulated node 
   back up to the root based on the simulation outcome.

The agent selects the path with the most visits after a fixed number of iterations.
"""
import math
import random
from typing import List, Optional

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj

# UCB1 exploration constant (controls exploration vs. exploitation)
# A common value is sqrt(2). Higher values encourage more exploration.
C_PUCT = math.sqrt(2) 

# Maximum depth for the simulation (rollout) phase.
# This prevents simulations from running indefinitely in non-terminal game loops.
MAX_ROLLOUT_DEPTH = 50 

class MCTSNode:
    """
    Represents a node in the Monte Carlo Search Tree (MCTS).

    Each node stores its game state, relationship to parent/children in the tree,
    statistics (wins and visits) for UCB1 calculation, the action that led to this node,
    and information about its terminal status and explored actions.
    """
    def __init__(self, state: GameState, parent: Optional['MCTSNode'], action: Optional[Direction]):
        self.state: GameState = state
        self.parent: Optional['MCTSNode'] = parent
        self.action: Optional[Direction] = action  # Action that led from parent to this node
        self.children: List['MCTSNode'] = []
        
        self.wins: float = 0.0
        self.visits: int = 0
        
        self.is_terminal_state: bool = False

        self.state: GameState = state
        self.parent: Optional['MCTSNode'] = parent
        self.action: Optional[Direction] = action  # Action that led from parent to this node
        self.children: List['MCTSNode'] = []
        
        self.wins: float = 0.0
        self.visits: int = 0
        
        self.is_terminal_state: bool = False
        self.update_terminal_status()

        self.untried_actions: List[Direction] = []
        if not self.is_terminal_state:
            self.untried_actions = self._get_possible_actions()

    def _get_possible_actions(self) -> List[Direction]:
        """
        Returns a list of all possible actions from this node's state.
        For Baba Is You, all cardinal directions plus 'Wait' are generally considered possible.
        The game engine (`advance_game_state`) will handle the consequences of actions,
        such as moving into a wall if no 'STOP' rule applies to it.
        """
        return [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]

    def update_terminal_status(self):
        """
        Checks and updates the `is_terminal_state` flag for this node.
        A state is considered terminal if a win condition is met (`check_win`)
        or if there are no controllable player objects left.
        """
        if check_win(self.state):
            self.is_terminal_state = True
        elif not self.state.players:  # No players left implies a loss or dead-end
            self.is_terminal_state = True
        else:
            self.is_terminal_state = False

    def is_fully_expanded(self) -> bool:
        """
        Checks if all possible actions from this node have been tried (i.e., have corresponding children).
        """
        return not self.untried_actions

    def select_best_child(self, c_param: float = C_PUCT) -> 'MCTSNode':
        """
        Selects the child with the highest Upper Confidence Bound 1 (UCB1) score.
        UCB1 = (wins / visits) + C * sqrt(log(parent_visits) / visits)
        This balances exploitation (choosing known good moves) and exploration (trying less-visited moves).

        Args:
            c_param: The exploration constant C in the UCB1 formula.

        Returns:
            The child node with the highest UCB1 score.
            Returns a random child if multiple children have infinite scores (e.g., all unvisited).
        """
        best_child = None
        best_score = -float('inf')
        
        # Tie-breaking for multiple 'infinite' scores (unvisited children)
        # Can happen if multiple children have 0 visits. We want to pick one, often randomly.
        candidates_for_tie_break = []

        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children with a very high score (effectively infinite)
                # This ensures that unvisited children are selected for exploration.
                score = float('inf')
            else:
                exploit_term = child.wins / child.visits
                # self.visits is parent_visits for the child
                explore_term = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit_term + explore_term
            
            if score > best_score:
                best_score = score
                best_child = child
                candidates_for_tie_break = [child] # Reset candidates
            elif score == best_score: # Handle ties, especially for multiple float('inf')
                if best_child: # only add if best_child is not None
                    candidates_for_tie_break.append(child)

        # If there was a tie (especially among unvisited nodes), pick one randomly.
        if len(candidates_for_tie_break) > 1 :
            return random.choice(candidates_for_tie_break)
        
        return best_child if best_child else (random.choice(self.children) if self.children else None)


class MCTSAgent(BaseAgent):
    """
    Agent that uses Monte Carlo Tree Search (MCTS) to find a sequence of actions
    to solve Baba Is You puzzles. It iteratively builds a search tree, balancing
    exploration of new paths with exploitation of known good paths.
    """
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Performs MCTS for a given number of iterations to find a solution path.

        Args:
            initial_state: The starting GameState of the puzzle.
            iterations: The number of MCTS iterations (selection, expansion, simulation, backpropagation cycles) to perform.

        Returns:
            A list of Direction enums representing the sequence of actions to reach a solution.
            Returns an empty list if no solution is found or if the initial state is invalid.
        """
        if not initial_state.players:
            # If there are no player objects, no actions can be taken.
            return [] 

        root_node = MCTSNode(state=initial_state.copy(), parent=None, action=None)

        # Handle edge cases: initial state is already a solution or an immediate dead-end.
        if root_node.is_terminal_state and check_win(root_node.state): # Solved at root
            return [] # No actions needed if already won
        if root_node.is_terminal_state: # Unsolvable from start (terminal but not a win)
             return []

        for _ in range(iterations):
            # 1. Selection phase: Traverse the tree from the root using UCB1.
            # Stop at a node that is either not fully expanded or is a terminal state.
            current_selection = root_node
            while not current_selection.is_terminal_state and \
                  current_selection.is_fully_expanded() and \
                  current_selection.children: # Must have children to select from
                current_selection = current_selection.select_best_child()
            
            # current_selection is now the node to potentially expand or simulate from.

            # 2. Expansion phase: If the selected node is not terminal and not fully expanded,
            # create one new child node by trying an untried action.
            node_for_simulation = current_selection 
            if not current_selection.is_terminal_state: 
                if not current_selection.is_fully_expanded(): 
                    expanded_child = self._expand(current_selection)
                    if expanded_child:
                        node_for_simulation = expanded_child
                # If fully expanded (and not terminal), simulation runs from current_selection.
                # This happens if selection led to an existing leaf that was already fully expanded
                # but whose children haven't been fully explored down another path.
            
            # 3. Simulation phase: From the 'node_for_simulation' (either newly expanded or selected leaf),
            # run a random playout until a terminal state or max depth is reached.
            reward: float
            if node_for_simulation.is_terminal_state:
                # If the node itself is terminal, its outcome is known.
                reward = 1.0 if check_win(node_for_simulation.state) else 0.0
            else:
                # Otherwise, simulate from this node's state.
                # Crucial: state must be copied for simulation to not affect the tree.
                reward = self._simulate(node_for_simulation.state.copy())

            # 4. Backpropagation phase: Update visit counts and win scores from the
            # 'node_for_simulation' back up to the root node.
            self._backpropagate(node_for_simulation, reward)

        # After all iterations, extract the best path from the root.
        # The best path is typically the one with the most visited children.
        best_path: List[Direction] = []
        current_node = root_node
        while current_node and current_node.children and not current_node.is_terminal_state:
            most_visited_child = None
            max_visits = -1
            for child in current_node.children:
                if child.visits > max_visits:
                    max_visits = child.visits
                    most_visited_child = child
            
            if most_visited_child and most_visited_child.action:
                best_path.append(most_visited_child.action)
                current_node = most_visited_child
            else:
                # No more children to explore or stuck.
                break 
        
        return best_path

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expands the given node by creating one new child node from an untried action.

        Args:
            node: The MCTSNode to expand.

        Returns:
            The newly created child MCTSNode, or None if expansion is not possible
            (e.g., node has no untried actions, though this check is usually done before calling).
        """
        if not node.untried_actions:
            # This case should ideally be prevented by checks in the main loop.
            return None 

        action_to_try = node.untried_actions.pop(0) # Try actions in a consistent order (FIFO)
        
        # Perform the action on a *copy* of the parent's state to get the new state.
        new_game_state = advance_game_state(action_to_try, node.state.copy()) 
        
        # Create the new child node.
        child_node = MCTSNode(state=new_game_state, parent=node, action=action_to_try)
        node.children.append(child_node)
        
        return child_node

    def _simulate(self, state: GameState) -> float:
        """
        Performs a random simulation (rollout) from the given game state.
        The simulation runs until a terminal state (win/loss) is reached or
        until `MAX_ROLLOUT_DEPTH` is exceeded.

        Args:
            state: The GameState to start the simulation from. This state should be a copy
                   and can be modified during simulation.

        Returns:
            A reward: 1.0 for a win, 0.0 for a loss or if max depth is reached without a win.
        """
        current_state = state # This state is already a copy from the caller.
        for _ in range(MAX_ROLLOUT_DEPTH):
            if check_win(current_state):
                return 1.0  # Win
            if not current_state.players: # No players left, considered a loss.
                return 0.0 

            # Choose a random action for the rollout.
            possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
            random_action = random.choice(possible_actions)
            
            # advance_game_state modifies the state.
            current_state = advance_game_state(random_action, current_state) 

        return 0.0 # Max depth reached without win/loss, considered a non-win.

    def _backpropagate(self, node: Optional[MCTSNode], reward: float):
        """
        Backpropagates the simulation result (reward) up the tree from the given node.
        Updates the `visits` and `wins` statistics for each node in the path to the root.

        Args:
            node: The MCTSNode from which the simulation was run (or its terminal child).
            reward: The reward obtained from the simulation (e.g., 1.0 for win, 0.0 for loss).
        """
        temp_node = node
        while temp_node:
            temp_node.visits += 1
            temp_node.wins += reward
            temp_node = temp_node.parent

# Helper for testing if needed, not part of the agent class.
# This allows for potential standalone testing of the agent if game setup code is added.
if __name__ == '__main__':
    # Example of how one might test this agent programmatically (actual execution
    # is usually handled by the KekeAI framework's `execution.py`).
    #
    # from baba import make_level, parse_map 
    # test_map_ascii = "__________\n_B12..F13_\n_........_\n_.b....f._\n__________"
    # game_map = parse_map(test_map_ascii)
    # initial_gs = make_level(game_map)
    #
    # agent = MCTSAgent()
    # print(f"Testing MCTSAgent on a demo level with, e.g., 1000 iterations...")
    # solution = agent.search(initial_gs, iterations=1000)
    # print(f"Solution found: {solution}")
    # print(f"Solution length: {len(solution)}")
    pass
