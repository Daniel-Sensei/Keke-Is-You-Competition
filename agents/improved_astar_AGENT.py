"""
Improved A* Agent for KekeAI Game with Enhanced Heuristics.

This enhanced version includes:
1. Rule priority system (YOU > WIN > others)
2. Detection of transformation chains and blocking rules
3. Multi-objective heuristic with rule interaction analysis
4. Dynamic penalty adjustment based on game state complexity
5. Better exploration of rule combinations
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObj
from typing import List, Set, Tuple, Dict
import heapq
from collections import defaultdict


class IMPROVED_ASTARAgent(BaseAgent):
    """
    Enhanced A* Search Agent with sophisticated rule-based heuristics.
    """

    # --- Enhanced Heuristic Parameters ---
    # Rule priority weights
    WEIGHT_YOU_RULE = 5.0           # Highest priority - player control
    WEIGHT_WIN_RULE = 4.0           # Second priority - win condition
    WEIGHT_TRANSFORMATION = 3.0      # Transformation rules
    WEIGHT_PROPERTY = 2.0           # Property rules (STOP, PUSH, etc.)
    WEIGHT_BLOCKING = 1.5           # Blocking/protective rules
    
    # Dynamic penalties
    BASE_PENALTY_MISSING_RULE = 25.0
    PENALTY_RULE_CONFLICT = 40.0     # When rules contradict each other
    PENALTY_UNREACHABLE_GOAL = 50.0  # When goal seems unreachable
    PENALTY_TRANSFORMATION_LOOP = 35.0 # Infinite transformation cycles
    
    # Distance weights
    WEIGHT_DISTANCE = 1.0
    WEIGHT_RULE_FORMATION = 2.5
    WEIGHT_RULE_INTERACTION = 1.8
    
    def __init__(self):
        super().__init__()
        self.rule_importance_cache = {}
        self.transformation_graph = {}

    def _analyze_rule_dependencies(self, state: GameState) -> Dict[str, float]:
        """
        Analyzes which rules are most critical for winning.
        Returns importance scores for different rule types.
        """
        importance = defaultdict(float)
        
        # Check what objects exist on the map
        object_types = set()
        if state.phys:
            for obj in state.phys:
                if hasattr(obj, 'name'):
                    object_types.add(obj.name.replace('_obj', ''))
        
        # Priority 1: YOU rule (player control is essential)
        has_you_rule = any('you' in rule for rule in state.rules)
        if not has_you_rule:
            # Find which objects could be players
            for obj_type in object_types:
                if obj_type in ['baba', 'keke']:  # Common player objects
                    importance[f"{obj_type}_is_you"] = self.WEIGHT_YOU_RULE
        
        # Priority 2: WIN rule (need a win condition)
        has_win_rule = any('win' in rule for rule in state.rules)
        if not has_win_rule:
            for obj_type in object_types:
                if obj_type in ['flag', 'love']:  # Common win objects
                    importance[f"{obj_type}_is_win"] = self.WEIGHT_WIN_RULE
        
        # Priority 3: Transformation rules that help reach goals
        if has_you_rule and has_win_rule:
            # Analyze if transformations are needed to reach win objects
            for obj_type in object_types:
                importance[f"transformation_{obj_type}"] = self.WEIGHT_TRANSFORMATION
        
        return dict(importance)

    def _detect_rule_conflicts(self, state: GameState) -> float:
        """
        Detects conflicting rules and calculates penalty.
        """
        penalty = 0.0
        rules = set(state.rules)
        
        # Check for direct conflicts like "X IS Y" and "X IS Z" where Y != Z
        transformation_rules = defaultdict(list)
        for rule in rules:
            parts = rule.split()
            if len(parts) == 3 and parts[1] == 'is':
                subject, verb, obj = parts
                if obj not in ['you', 'win', 'stop', 'push', 'hot', 'melt', 'kill', 'sink', 'move']:
                    transformation_rules[subject].append(obj)
        
        # Penalty for conflicting transformations
        for subject, targets in transformation_rules.items():
            if len(targets) > 1:
                penalty += self.PENALTY_RULE_CONFLICT
        
        # Check for blocking rules like "X IS X" that prevent needed transformations
        reflexive_rules = {rule.split()[0] for rule in rules if len(rule.split()) == 3 and rule.split()[0] == rule.split()[2]}
        for blocked_obj in reflexive_rules:
            if any(blocked_obj in rule and blocked_obj != rule.split()[2] for rule in rules):
                penalty += self.PENALTY_RULE_CONFLICT * 0.5
        
        return penalty

    def _estimate_transformation_path(self, state: GameState) -> float:
        """
        Estimates the cost of transformation chains needed to win.
        """
        if not state.players or not state.winnables:
            return 0.0
        
        # Get player and win object types
        player_types = {p.name.replace('_obj', '') for p in state.players if hasattr(p, 'name')}
        win_types = {w.name.replace('_obj', '') for w in state.winnables if hasattr(w, 'name')}
        
        # If player can already reach win objects directly, no transformation needed
        min_direct_distance = float('inf')
        for player in state.players:
            for win_obj in state.winnables:
                dist = abs(player.x - win_obj.x) + abs(player.y - win_obj.y)
                min_direct_distance = min(min_direct_distance, dist)
        
        if min_direct_distance < 5:  # Close enough for direct approach
            return 0.0
        
        # Check if transformations could help
        transformation_cost = 0.0
        
        # Look for beneficial transformations
        object_positions = defaultdict(list)
        if state.phys:
            for obj in state.phys:
                if hasattr(obj, 'name'):
                    obj_type = obj.name.replace('_obj', '')
                    object_positions[obj_type].append((obj.x, obj.y))
        
        # Estimate cost of transforming objects to create better paths
        for obj_type, positions in object_positions.items():
            if obj_type not in player_types and obj_type not in win_types:
                # Could this object be transformed into something useful?
                for pos in positions:
                    for win_pos in [(w.x, w.y) for w in state.winnables]:
                        if abs(pos[0] - win_pos[0]) + abs(pos[1] - win_pos[1]) < min_direct_distance:
                            transformation_cost += self._estimate_rule_formation_cost(
                                state, obj_type, "is", list(win_types)[0] if win_types else "flag"
                            )
                            break
        
        return min(transformation_cost, self.PENALTY_UNREACHABLE_GOAL)

    def _calculate_enhanced_heuristic(self, state: GameState) -> float:
        """
        Enhanced heuristic function that considers rule priorities and interactions.
        """
        if check_win(state):
            return 0.0
        
        total_cost = 0.0
        
        # 1. Basic distance component
        distance_cost = self._heuristic_distance(state)
        total_cost += self.WEIGHT_DISTANCE * distance_cost
        
        # 2. Rule formation cost with priorities
        rule_importance = self._analyze_rule_dependencies(state)
        rule_cost = 0.0
        
        for rule_pattern, importance in rule_importance.items():
            if '_is_' in rule_pattern:
                parts = rule_pattern.split('_is_')
                if len(parts) == 2:
                    formation_cost = self._estimate_rule_formation_cost(state, parts[0], "is", parts[1])
                    rule_cost += importance * formation_cost
        
        total_cost += self.WEIGHT_RULE_FORMATION * rule_cost
        
        # 3. Rule conflict detection
        conflict_penalty = self._detect_rule_conflicts(state)
        total_cost += conflict_penalty
        
        # 4. Transformation path analysis
        transformation_cost = self._estimate_transformation_path(state)
        total_cost += self.WEIGHT_RULE_INTERACTION * transformation_cost
        
        # 5. Dynamic penalty based on game complexity
        complexity_factor = len(state.rules) + len(state.phys or []) / 10.0
        total_cost *= (1.0 + complexity_factor * 0.1)
        
        return total_cost

    def _estimate_rule_formation_cost(self, state: GameState, term1_base: str, term2_base_is: str, term3_base: str) -> float:
        """
        Enhanced rule formation cost estimation with better word finding.
        """
        term1_objs = self._get_word_objects_by_base_name(state, term1_base)
        is_objs = self._get_word_objects_by_base_name(state, term2_base_is) 
        term3_objs = self._get_word_objects_by_base_name(state, term3_base)

        if not term1_objs or not is_objs or not term3_objs:
            return self.BASE_PENALTY_MISSING_RULE

        min_cost = float('inf')

        for is_obj in is_objs:
            is_x, is_y = is_obj.x, is_obj.y
            
            # Horizontal formation: T1 - IS - T3
            for t1 in term1_objs:
                for t3 in term3_objs:
                    cost_h = (abs(t1.x - (is_x - 1)) + abs(t1.y - is_y) + 
                             abs(t3.x - (is_x + 1)) + abs(t3.y - is_y))
                    min_cost = min(min_cost, cost_h)
            
            # Vertical formation: T1 / IS / T3
            for t1 in term1_objs:
                for t3 in term3_objs:
                    cost_v = (abs(t1.x - is_x) + abs(t1.y - (is_y - 1)) + 
                             abs(t3.x - is_x) + abs(t3.y - (is_y + 1)))
                    min_cost = min(min_cost, cost_v)
        
        return min_cost if min_cost != float('inf') else self.BASE_PENALTY_MISSING_RULE

    def _get_word_objects_by_base_name(self, state: GameState, base_name: str) -> List[GameObj]:
        """Enhanced word object finder."""
        found_objects = []
        for word_obj_list in [state.words, state.keywords]:
            if word_obj_list:
                for obj in word_obj_list:
                    if hasattr(obj, 'name') and obj.name == base_name:
                        found_objects.append(obj)
        return found_objects

    def _get_state_hash(self, game_state: GameState) -> tuple:
        """Enhanced state hashing that includes rule dependencies."""
        phys_tuples = []
        if game_state.phys:
            for p in sorted([obj for obj in game_state.phys if hasattr(obj, 'name')], 
                          key=lambda obj: (obj.name, obj.x, obj.y)):
                phys_tuples.append((p.name, p.x, p.y))
        
        word_tuples = []
        all_word_objects = []
        if game_state.words:
            all_word_objects.extend([obj for obj in game_state.words if hasattr(obj, 'name')])
        if game_state.keywords:
            all_word_objects.extend([obj for obj in game_state.keywords if hasattr(obj, 'name')])
            
        if all_word_objects:
            for w in sorted(all_word_objects, key=lambda obj: (obj.name, obj.x, obj.y)):
                word_tuples.append((w.name, w.x, w.y))
        
        rules_tuple = tuple(sorted(list(set(game_state.rules))))
        
        # Include rule interaction state
        rule_priorities = self._analyze_rule_dependencies(game_state)
        priority_tuple = tuple(sorted(rule_priorities.items()))
        
        return (tuple(phys_tuples), tuple(word_tuples), rules_tuple, priority_tuple)

    def _heuristic_distance(self, state: GameState) -> float:
        """Enhanced distance heuristic."""
        if check_win(state):
            return 0.0

        if not state.players:
            return self.BASE_PENALTY_MISSING_RULE
        
        has_win_rule = any("win" in rule for rule in state.rules)
        if not has_win_rule:
            return 0.0
        
        if not state.winnables:
            return self.BASE_PENALTY_MISSING_RULE

        # Calculate minimum distance with path quality consideration
        min_dist = float('inf')
        for player_obj in state.players:
            for winnable_obj in state.winnables:
                dist = abs(player_obj.x - winnable_obj.x) + abs(player_obj.y - winnable_obj.y)
                
                # Bonus for clear paths (simplified check)
                path_bonus = 0
                if dist < 3:  # Close objects get bonus
                    path_bonus = -2
                
                min_dist = min(min_dist, dist + path_bonus)
        
        return max(0, min_dist) if min_dist != float('inf') else self.BASE_PENALTY_MISSING_RULE

    def search(self, initial_state: GameState, iterations: int = 100) -> List[Direction]:
        """
        Enhanced A* search with better heuristics and increased iteration limit.
        """
        open_set: List[Tuple[float, float, tuple, GameState, List[Direction]]] = []
        closed_set: Set[tuple] = set()
        g_costs: Dict[tuple, float] = {}

        initial_state_hash = self._get_state_hash(initial_state)
        g_initial = 0.0
        h_initial = self._calculate_enhanced_heuristic(initial_state)
        
        if h_initial == float('inf'):
            return []
            
        f_initial = g_initial + h_initial

        heapq.heappush(open_set, (f_initial, g_initial, initial_state_hash, initial_state, []))
        g_costs[initial_state_hash] = g_initial
        
        expanded_nodes_count = 0
        possible_actions = [d for d in Direction if d != Direction.Undefined]

        while open_set and expanded_nodes_count < iterations:
            current_f, current_g, current_hash, current_state, current_path = heapq.heappop(open_set)

            if current_hash in closed_set:
                continue 
            
            closed_set.add(current_hash)
            expanded_nodes_count += 1

            if check_win(current_state):
                return current_path

            # Enhanced neighbor exploration with action prioritization
            action_priorities = self._prioritize_actions(current_state, possible_actions)
            
            for action in action_priorities:
                neighbor_state = advance_game_state(action, current_state.copy())
                neighbor_hash = self._get_state_hash(neighbor_state)
                
                new_g_cost = current_g + 1.0

                if new_g_cost >= g_costs.get(neighbor_hash, float('inf')):
                    continue
                
                g_costs[neighbor_hash] = new_g_cost
                h_cost = self._calculate_enhanced_heuristic(neighbor_state)

                if h_cost == float('inf'): 
                    continue 

                f_cost = new_g_cost + h_cost
                new_path = current_path + [action]
                heapq.heappush(open_set, (f_cost, new_g_cost, neighbor_hash, neighbor_state, new_path))
                
        return []

    def _prioritize_actions(self, state: GameState, actions: List[Direction]) -> List[Direction]:
        """
        Prioritizes actions based on game state analysis.
        """
        if not state.players:
            return actions
        
        # Simple prioritization: favor moves toward goals or rule formation
        player = state.players[0]
        
        # If we have both player and win objects, prioritize moves toward win
        if state.winnables:
            win_obj = state.winnables[0]
            preferred_actions = []
            
            if win_obj.x > player.x:
                preferred_actions.append(Direction.Right)
            elif win_obj.x < player.x:
                preferred_actions.append(Direction.Left)
                
            if win_obj.y > player.y:
                preferred_actions.append(Direction.Down)
            elif win_obj.y < player.y:
                preferred_actions.append(Direction.Up)
            
            # Add remaining actions
            remaining = [a for a in actions if a not in preferred_actions]
            return preferred_actions + remaining
        
        return actions
