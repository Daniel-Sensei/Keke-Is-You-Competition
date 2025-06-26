"""
A* Agent for KekeAI Game.

This agent implements the A* search algorithm to find solutions for puzzles
in a "Baba Is You" like environment. It uses a heuristic function that combines
Manhattan distance to goal objects with an estimation of the cost to form
essential game rules (e.g., "BABA IS YOU", "FLAG IS WIN").

Key Components:
- A* Search Loop: Manages an open set (priority queue) and a closed set (hashed states)
  to explore the game state space efficiently.
- Heuristic Function (`_calculate_heuristic`):
    - Distance Component (`_heuristic_distance`): Calculates Manhattan distance
      to winnable objects if player-controlled objects and win conditions are defined.
    - Rule Component (`_heuristic_rules`): Estimates the cost to form crucial
      rules like "X IS YOU" and "Y IS WIN" by calculating the sum of Manhattan
      distances required to move the constituent text objects into alignment.
      Applies penalties if essential objects (player, winnables) or words for rules
      are missing.
- State Hashing (`_get_state_hash`): Uses a tuple-based representation of key game
  elements (object positions, active rules) for efficient state tracking in the closed set.
- Configurable Parameters: Heuristic weights and penalties can be tuned. The search
  is capped by an iteration limit (node expansion budget).
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj # Added GameObj for type hinting
from typing import List, Set, Tuple, Dict # Added Dict for type hinting
import heapq
# from tqdm import trange # tqdm not used in current implementation


class ASTARAgent(BaseAgent):
    """
    A* Search Agent that uses a composite heuristic (distance + rule formation cost)
    to find solutions in the KekeAI game environment.
    """

    # --- Heuristic Parameters ---
    # These constants define penalties and weights for the heuristic function.
    # Tuning these values can significantly impact the agent's performance.

    PENALTY_NO_PLAYER_CONTROL = 30.0    # Base penalty if no "X IS YOU" rule is active and words are missing.
    PENALTY_NO_WIN_CONDITION_RULE = 30.0 # Base penalty if no "Y IS WIN" rule is active and words are missing.
    PENALTY_NO_WINNABLE_OBJECTS = 20.0   # Penalty if a "Y IS WIN" rule exists, but no Y objects are on map.
    PENALTY_NO_PLAYER_OBJECTS = 20.0     # Penalty if an "X IS YOU" rule exists, but no X objects are on map.
    PENALTY_WORD_TYPE_MISSING = 10.0     # Cost added to heuristic's rule formation estimate if a word (e.g., "BABA") is not on map.
    
    HEURISTIC_WEIGHT_DISTANCE = 1.0     # Weight for the distance component of the heuristic.
    HEURISTIC_WEIGHT_RULES = 1.0        # Weight for the rule-based component.

    # HEURISTIC_MAX_DEAD_END = float('inf') # Not explicitly used, but individual heuristic components can return float('inf') or large numbers.


    def _get_word_objects_by_base_name(self, state: GameState, base_name: str) -> List[GameObj]:
        """
        Helper function to retrieve all GameObj instances of words (text objects)
        that match a given base name (e.g., "baba", "is", "flag", "win").
        It checks both `state.words` (for object words like BABA_WORD) and
        `state.keywords` (for keywords like IS_WORD, YOU_WORD, WIN_WORD).

        Args:
            state: The current GameState.
            base_name: The base name of the word to search for (e.g., "baba").

        Returns:
            A list of GameObj instances matching the base name.
        """
        found_objects = []
        # GameObj.name for text objects stores the base name (e.g., "baba" for BABA_WORD).
        for word_obj_list in [state.words, state.keywords]:
            if word_obj_list:
                for obj in word_obj_list:
                    if hasattr(obj, 'name') and obj.name == base_name:
                        found_objects.append(obj)
        return found_objects

    def _estimate_rule_formation_cost(self, state: GameState, term1_base: str, term2_base_is: str, term3_base: str) -> float:
        """
        Estimates the minimum sum of Manhattan distances required to move three specified
        text objects (term1, IS, term3) into alignment to form a rule.
        Considers all instances of these words on the map and both horizontal and
        vertical rule formations around each 'IS' word.

        Example: For "BABA IS YOU", term1_base="baba", term2_base_is="is", term3_base="you".

        Args:
            state: The current GameState.
            term1_base: Base name of the first word object (e.g., "baba").
            term2_base_is: Base name of the 'IS' word object (should always be "is").
            term3_base: Base name of the third word object (e.g., "you").

        Returns:
            The estimated Manhattan distance cost, or `self.PENALTY_WORD_TYPE_MISSING`
            if any of the required word types are not present on the map.
        """
        term1_objs = self._get_word_objects_by_base_name(state, term1_base)
        is_objs = self._get_word_objects_by_base_name(state, term2_base_is) 
        term3_objs = self._get_word_objects_by_base_name(state, term3_base)

        if not term1_objs or not is_objs or not term3_objs:
            return self.PENALTY_WORD_TYPE_MISSING 

        min_overall_formation_cost = float('inf')

        for is_obj in is_objs:
            is_x, is_y = is_obj.x, is_obj.y
            
            # Cost for Horizontal formation: T1 - IS - T3
            # Target for T1: (is_x - 1, is_y), Target for T3: (is_x + 1, is_y)
            cost_t1_h, cost_t3_h = float('inf'), float('inf')
            
            # Find closest term1_obj for horizontal
            for t1 in term1_objs:
                cost_t1_h = min(cost_t1_h, abs(t1.x - (is_x - 1)) + abs(t1.y - is_y))
            # Find closest term3_obj for horizontal
            for t3 in term3_objs:
                cost_t3_h = min(cost_t3_h, abs(t3.x - (is_x + 1)) + abs(t3.y - is_y))
            
            if cost_t1_h != float('inf') and cost_t3_h != float('inf'):
                min_overall_formation_cost = min(min_overall_formation_cost, cost_t1_h + cost_t3_h)

            # Cost for Vertical formation: T1 / IS / T3
            # Target for T1: (is_x, is_y - 1), Target for T3: (is_x, is_y + 1)
            cost_t1_v, cost_t3_v = float('inf'), float('inf')

            # Find closest term1_obj for vertical
            for t1 in term1_objs:
                cost_t1_v = min(cost_t1_v, abs(t1.x - is_x) + abs(t1.y - (is_y - 1)))
            # Find closest term3_obj for vertical
            for t3 in term3_objs:
                cost_t3_v = min(cost_t3_v, abs(t3.x - is_x) + abs(t3.y - (is_y + 1)))

            if cost_t1_v != float('inf') and cost_t3_v != float('inf'):
                min_overall_formation_cost = min(min_overall_formation_cost, cost_t1_v + cost_t3_v)
        
        return min_overall_formation_cost if min_overall_formation_cost != float('inf') else self.PENALTY_WORD_TYPE_MISSING


    def _get_state_hash(self, game_state: GameState) -> tuple:
        """
        Creates a hashable tuple representing the core components of the game state.
        Used for the closed set in A*.
        """
        phys_tuples = []
        if game_state.phys:
            for p in sorted([obj for obj in game_state.phys if hasattr(obj, 'name')], key=lambda obj: (obj.name, obj.x, obj.y)):
                phys_tuples.append((p.name, p.x, p.y))
        
        word_tuples = []
        all_word_like_objects = []
        if game_state.words:
            all_word_like_objects.extend([obj for obj in game_state.words if hasattr(obj, 'name')])
        if game_state.keywords:
            all_word_like_objects.extend([obj for obj in game_state.keywords if hasattr(obj, 'name')])
            
        if all_word_like_objects:
            for w in sorted(all_word_like_objects, key=lambda obj: (obj.name, obj.x, obj.y)):
                word_tuples.append((w.name, w.x, w.y))
        
        rules_tuple = tuple(sorted(list(set(game_state.rules))))
        return (tuple(phys_tuples), tuple(word_tuples), rules_tuple)

    def _heuristic_distance(self, state: GameState) -> float:
        """
        Calculates the distance component of the heuristic.
        Estimates cost based on Manhattan distance between player(s) and winnable(s).
        Returns 0 if already in a win state.
        Returns high penalties if no player objects exist or if winnable objects are
        expected (due to a WIN rule) but none are present.
        If no WIN rule is active, distance to winnables is considered 0 as it's not relevant yet.
        """
        if check_win(state):
            return 0.0

        if not state.players:
            # This state is dire: no player objects, even if a "YOU" rule might exist.
            # The rule heuristic will heavily penalize lack of "YOU" rule or missing player objects.
            # Distance is somewhat meaningless here, but assign high cost.
            return self.PENALTY_NO_PLAYER_OBJECTS 
        
        has_win_rule = any("win" in rule for rule in state.rules)
        if not has_win_rule:
            # If there's no "Y IS WIN" rule, distance to a winnable object is not a meaningful heuristic yet.
            # The rule heuristic will penalize the lack of a win condition rule.
            return 0.0
        
        if not state.winnables:
            # A "Y IS WIN" rule exists, but no Y objects are currently on the map.
            return self.PENALTY_NO_WINNABLE_OBJECTS 

        min_dist = float('inf')
        for player_obj in state.players:
            for winnable_obj in state.winnables:
                dist = abs(player_obj.x - winnable_obj.x) + abs(player_obj.y - winnable_obj.y)
                min_dist = min(min_dist, dist)
        
        # This case should ideally be covered by 'if not state.winnables' if min_dist remains inf.
        return min_dist if min_dist != float('inf') else self.PENALTY_NO_WINNABLE_OBJECTS

    def _heuristic_rules(self, state: GameState) -> float:
        """
        Calculates the rule-based component of the heuristic.
        If essential rules ("X IS YOU", "Y IS WIN") are not active, it estimates
        the Manhattan distance cost to arrange the corresponding text objects to form these rules.
        Uses default rules "BABA IS YOU" and "FLAG IS WIN" as proxies if specific object names
        are not easily determinable as the primary player/win condition from the state.
        Penalties are applied if necessary word objects are missing from the map,
        or if rules are active but corresponding game objects (player, winnables) are not present.
        """
        cost = 0.0
        rules_set = set(state.rules)

        # --- Player Control Heuristic ("X IS YOU") ---
        player_control_active = any("you" in r for r in rules_set)
        
        if not player_control_active:
            # Estimate cost to form "BABA IS YOU" as a common default.
            # A more sophisticated version could try to identify which object X in "X IS YOU"
            # would be most beneficial or easiest to form.
            formation_cost_baba_is_you = self._estimate_rule_formation_cost(state, "baba", "is", "you")
            if formation_cost_baba_is_you >= self.PENALTY_WORD_TYPE_MISSING:
                # If words for "BABA IS YOU" are missing, apply a large flat penalty.
                cost += self.PENALTY_NO_PLAYER_CONTROL 
            else:
                cost += formation_cost_baba_is_you
        elif not state.players: 
            # A "YOU" rule exists, but no corresponding player objects (e.g., BABA_OBJ) are on map.
            cost += self.PENALTY_NO_PLAYER_OBJECTS 

        # --- Win Condition Heuristic ("Y IS WIN") ---
        win_condition_active = any("win" in r for r in rules_set)

        if not win_condition_active:
            # Estimate cost to form "FLAG IS WIN" as a common default.
            formation_cost_flag_is_win = self._estimate_rule_formation_cost(state, "flag", "is", "win")
            if formation_cost_flag_is_win >= self.PENALTY_WORD_TYPE_MISSING:
                cost += self.PENALTY_NO_WIN_CONDITION_RULE
            else:
                cost += formation_cost_flag_is_win
        elif not state.winnables:
            # A "WIN" rule exists, but no corresponding winnable objects (e.g., FLAG_OBJ) are on map.
            cost += self.PENALTY_NO_WINNABLE_OBJECTS
            
        return cost

    def _calculate_heuristic(self, state: GameState) -> float:
        """
        The main heuristic function h(n) for the A* search.
        It combines a distance-based component and a rule-based component,
        weighted by `HEURISTIC_WEIGHT_DISTANCE` and `HEURISTIC_WEIGHT_RULES`.
        Returns 0 if the state is already a win state.
        The components can return large penalty values if critical game elements
        (player control, win conditions, necessary objects) are missing or unachievable.
        """
        if check_win(state): # Goal state has h=0
            return 0.0
        
        rule_cost = self._heuristic_rules(state)
        distance_cost = self._heuristic_distance(state)
        
        # If rule_cost is extremely high (e.g., indicating missing words for essential rules),
        # it might dominate. The weighted sum naturally handles this.
        # No special capping here unless specific issues arise.

        h_val = (self.HEURISTIC_WEIGHT_DISTANCE * distance_cost +
                 self.HEURISTIC_WEIGHT_RULES * rule_cost)
        
        return h_val

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        Implements the A* search algorithm to find an optimal path to a win state.

        The search uses a priority queue (open set) ordered by f_cost (g_cost + h_cost),
        a closed set to track visited states (using a tuple-based hash), and a dictionary
        to store the minimum g_costs found to reach each state. The `iterations` parameter
        serves as a budget for the number of node expansions to prevent excessively long searches.

        Args:
            initial_state: The starting GameState of the puzzle.
            iterations: Maximum number of nodes to expand from the open set.

        Returns:
            A list of Direction enums representing the sequence of actions to reach a solution,
            or an empty list if no solution is found within the iteration budget.
        """
        # (f_cost, g_cost, state_hash, GameState object, path_list_of_Directions)
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        
        # Stores state_hash of expanded nodes
        closed_set: Set[tuple] = set()
        
        # Stores g_cost for states encountered: state_hash -> g_cost
        # Used to update paths if a cheaper way to a state in open_set is found,
        # or to avoid re-processing from closed_set if not better.
        g_costs: Dict[tuple, float] = {}

        initial_state_hash = self._get_state_hash(initial_state)
        g_initial = 0.0
        h_initial = self._calculate_heuristic(initial_state)
        
        # If initial heuristic is infinite, it's likely an unsolvable start from heuristic's PoV
        if h_initial == float('inf'):
            return []
            
        f_initial = g_initial + h_initial

        heapq.heappush(open_set, (f_initial, g_initial, initial_state_hash, initial_state, []))
        g_costs[initial_state_hash] = g_initial
        
        expanded_nodes_count = 0
        possible_actions = [d for d in Direction if d != Direction.Undefined] # Cache available actions

        while open_set and expanded_nodes_count < iterations:
            # Pop the node with the smallest f_cost from the priority queue
            current_f, current_g, current_hash, current_state, current_path = heapq.heappop(open_set)

            # If this state hash is already in closed_set, it means we've processed this state
            # via an equal or shorter path already. Skip.
            if current_hash in closed_set:
                continue 
            
            closed_set.add(current_hash)
            expanded_nodes_count += 1

            if check_win(current_state):
                return current_path # Solution found

            # Explore neighbors
            for action in possible_actions:
                # Create a deep copy for the neighbor state to avoid modifying current_state
                # or states already in the open/closed sets.
                neighbor_state = advance_game_state(action, current_state.copy())
                neighbor_hash = self._get_state_hash(neighbor_state)
                
                new_g_cost = current_g + 1.0 # Cost of each action is assumed to be 1

                # If already found a cheaper or equal path to this neighbor, skip
                if new_g_cost >= g_costs.get(neighbor_hash, float('inf')):
                    continue
                
                # Update g_cost for this neighbor and calculate heuristic
                g_costs[neighbor_hash] = new_g_cost
                h_cost = self._calculate_heuristic(neighbor_state)

                # If heuristic indicates a dead-end or impossibly high cost, prune this path
                if h_cost == float('inf'): 
                    continue 

                f_cost = new_g_cost + h_cost
                new_path = current_path + [action]
                heapq.heappush(open_set, (f_cost, new_g_cost, neighbor_hash, neighbor_state, new_path))
                
        return [] # No solution found within the iteration budget or open_set exhausted