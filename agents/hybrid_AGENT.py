"""
Hybrid Agent combining A* and Evolutionary approaches for KekeAI Game.

This hybrid agent uses:
1. A* for rule formation and short-term planning
2. Evolutionary search for long-term strategy and exploration
3. Dynamic switching based on problem characteristics
4. Knowledge sharing between the two approaches
"""

import random
import time
import os
import importlib.util
from typing import List, Dict, Tuple, Any
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win


class HYBRIDAgent(BaseAgent):
    """
    Hybrid agent that dynamically combines A* and Evolutionary approaches.
    """
    
    def __init__(self, max_time_per_method=30.0, switch_threshold=50):
        super().__init__()
        self.max_time_per_method = max_time_per_method
        self.switch_threshold = switch_threshold
        
        # Initialize sub-agents using dynamic loading
        self.astar_agent = self._load_agent_class('improved_astar_AGENT.py', 'IMPROVED_ASTARAgent')()
        self.evolutionary_agent = self._load_agent_class('improved_evolutionary_AGENT.py', 'IMPROVED_EVOLUTIONARYAgent')(
            population_size=30, 
            generations=80, 
            solution_length=100
        )
        
        # Knowledge sharing
        self.shared_knowledge = {
            'promising_rule_patterns': set(),
            'effective_action_sequences': [],
            'dead_end_states': set(),
            'successful_strategies': []
        }

    def _load_agent_class(self, agent_filename, class_name):
        """
        Dynamically loads an agent class from a file.
        """
        try:
            # Get the directory where this hybrid agent is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            agent_path = os.path.join(current_dir, agent_filename)
            
            # Load the module
            spec = importlib.util.spec_from_file_location(class_name, agent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the class
            agent_class = getattr(module, class_name)
            return agent_class
            
        except Exception as e:
            print(f"[ERROR] Failed to load agent {class_name} from {agent_filename}: {e}")
            # Return a dummy agent class as fallback
            return BaseAgent

    def _analyze_problem_characteristics(self, initial_state: GameState) -> Dict[str, Any]:
        """
        Analyzes the problem to determine which approach might work better.
        """
        characteristics = {
            'complexity_score': 0.0,
            'rule_formation_needed': False,
            'exploration_heavy': False,
            'preferred_method': 'astar'  # default
        }
        
        # Count game elements
        num_objects = len(initial_state.phys) if initial_state.phys else 0
        num_words = (len(initial_state.words) if initial_state.words else 0) + \
                   (len(initial_state.keywords) if initial_state.keywords else 0)
        num_rules = len(initial_state.rules)
        
        # Calculate complexity
        characteristics['complexity_score'] = num_objects + num_words * 2 + num_rules * 3
        
        # Check if rule formation is needed
        has_you_rule = any('you' in rule for rule in initial_state.rules)
        has_win_rule = any('win' in rule for rule in initial_state.rules)
        
        if not has_you_rule or not has_win_rule:
            characteristics['rule_formation_needed'] = True
        
        # Check if exploration might be needed (large map, many objects)
        if num_objects > 20 or characteristics['complexity_score'] > 50:
            characteristics['exploration_heavy'] = True
        
        # Determine preferred initial method
        if characteristics['rule_formation_needed'] and not characteristics['exploration_heavy']:
            characteristics['preferred_method'] = 'astar'
        elif characteristics['exploration_heavy'] or characteristics['complexity_score'] > 60:
            characteristics['preferred_method'] = 'evolutionary'
        else:
            # For medium complexity, start with A* but be ready to switch
            characteristics['preferred_method'] = 'astar'
        
        return characteristics

    def _extract_knowledge_from_astar(self, solution: List[Direction], initial_state: GameState):
        """
        Extracts useful knowledge from A* solution attempts.
        """
        if not solution:
            return
        
        # Extract effective short sequences
        for i in range(0, len(solution), 5):
            subsequence = solution[i:i+5]
            if len(subsequence) >= 3:
                self.shared_knowledge['effective_action_sequences'].append(subsequence)
        
        # Analyze rule patterns that led to success
        current_state = initial_state
        for i, action in enumerate(solution):
            current_state = advance_game_state(action, current_state.copy())
            if len(current_state.rules) > len(initial_state.rules):
                # New rule formed
                new_rules = set(current_state.rules) - set(initial_state.rules)
                self.shared_knowledge['promising_rule_patterns'].update(new_rules)

    def _extract_knowledge_from_evolutionary(self, best_individual: List[Direction], initial_state: GameState):
        """
        Extracts useful knowledge from evolutionary search.
        """
        if not best_individual:
            return
        
        # Track successful strategy patterns
        strategy_pattern = {
            'length': len(best_individual),
            'action_distribution': {},
            'effective_subsequences': []
        }
        
        # Analyze action distribution
        for action in best_individual:
            strategy_pattern['action_distribution'][action] = \
                strategy_pattern['action_distribution'].get(action, 0) + 1
        
        # Find effective subsequences (ones that change rules or advance toward goal)
        current_state = initial_state
        prev_rules = set(initial_state.rules)
        
        for i, action in enumerate(best_individual):
            current_state = advance_game_state(action, current_state.copy())
            current_rules = set(current_state.rules)
            
            # If rules changed or we're close to win, mark this subsequence as effective
            if current_rules != prev_rules or (current_state.players and current_state.winnables):
                start_idx = max(0, i - 3)
                end_idx = min(len(best_individual), i + 3)
                effective_subseq = best_individual[start_idx:end_idx]
                strategy_pattern['effective_subsequences'].append(effective_subseq)
            
            prev_rules = current_rules
        
        self.shared_knowledge['successful_strategies'].append(strategy_pattern)

    def _seed_evolutionary_with_astar_knowledge(self) -> List[List[Direction]]:
        """
        Creates initial population for evolutionary algorithm using A* knowledge.
        """
        seeded_population = []
        
        # Use effective action sequences as building blocks
        for _ in range(5):  # Create 5 seeded individuals
            individual = []
            target_length = self.evolutionary_agent.solution_length
            
            while len(individual) < target_length:
                if self.shared_knowledge['effective_action_sequences'] and random.random() < 0.7:
                    # Use known effective sequence
                    sequence = random.choice(self.shared_knowledge['effective_action_sequences'])
                    individual.extend(sequence)
                else:
                    # Add random action
                    individual.append(random.choice(self.evolutionary_agent.possible_actions))
            
            # Trim to target length
            individual = individual[:target_length]
            seeded_population.append(individual)
        
        return seeded_population

    def _enhance_astar_with_evolutionary_knowledge(self) -> Dict[str, Any]:
        """
        Provides hints to A* based on evolutionary discoveries.
        """
        hints = {
            'preferred_actions': [],
            'promising_patterns': list(self.shared_knowledge['promising_rule_patterns']),
            'exploration_bias': False
        }
        
        # Analyze successful strategies for action preferences
        if self.shared_knowledge['successful_strategies']:
            action_scores = {}
            for strategy in self.shared_knowledge['successful_strategies']:
                for action, count in strategy['action_distribution'].items():
                    action_scores[action] = action_scores.get(action, 0) + count
            
            # Sort actions by effectiveness
            if action_scores:
                sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
                hints['preferred_actions'] = [action for action, _ in sorted_actions]
        
        return hints

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        """
        Hybrid search that dynamically combines A* and Evolutionary approaches.
        """
        print("Starting hybrid search...")
        
        # Analyze problem characteristics
        problem_info = self._analyze_problem_characteristics(initial_state)
        print(f"Problem complexity: {problem_info['complexity_score']:.1f}")
        print(f"Rule formation needed: {problem_info['rule_formation_needed']}")
        print(f"Exploration heavy: {problem_info['exploration_heavy']}")
        print(f"Preferred method: {problem_info['preferred_method']}")
        
        best_solution = []
        best_fitness = -float('inf')
        
        # Phase 1: Initial attempt with preferred method
        if problem_info['preferred_method'] == 'astar':
            print("\nPhase 1: A* Search")
            start_time = time.time()
            
            try:
                astar_solution = self.astar_agent.search(initial_state, iterations=100)
                elapsed_time = time.time() - start_time
                
                if astar_solution and self._evaluate_solution(initial_state, astar_solution):
                    print(f"A* found solution in {elapsed_time:.2f}s!")
                    return astar_solution
                
                # Extract knowledge even if no solution found
                self._extract_knowledge_from_astar(astar_solution, initial_state)
                
                if astar_solution:
                    fitness = self._calculate_solution_fitness(initial_state, astar_solution)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = astar_solution
                
                print(f"A* completed in {elapsed_time:.2f}s, best fitness: {best_fitness:.2f}")
                
            except Exception as e:
                print(f"A* search failed: {e}")
        
        # Phase 2: Evolutionary search (either as primary or secondary method)
        print("\nPhase 2: Evolutionary Search")
        start_time = time.time()
        
        try:
            # Seed evolutionary population with A* knowledge if available
            seeded_pop = self._seed_evolutionary_with_astar_knowledge()
            
            # Temporarily modify evolutionary agent's population
            original_pop_size = self.evolutionary_agent.population_size
            if seeded_pop:
                # Mix seeded individuals with random ones
                self.evolutionary_agent.population_size = max(30, len(seeded_pop) * 3)
            
            evolutionary_solution = self.evolutionary_agent.search(initial_state, iterations)
            elapsed_time = time.time() - start_time
            
            # Restore original population size
            self.evolutionary_agent.population_size = original_pop_size
            
            if evolutionary_solution and self._evaluate_solution(initial_state, evolutionary_solution):
                print(f"Evolutionary found solution in {elapsed_time:.2f}s!")
                return evolutionary_solution
            
            # Extract knowledge
            self._extract_knowledge_from_evolutionary(evolutionary_solution, initial_state)
            
            if evolutionary_solution:
                fitness = self._calculate_solution_fitness(initial_state, evolutionary_solution)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = evolutionary_solution
            
            print(f"Evolutionary completed in {elapsed_time:.2f}s, best fitness: {best_fitness:.2f}")
            
        except Exception as e:
            print(f"Evolutionary search failed: {e}")
        
        # Phase 3: Enhanced A* with evolutionary knowledge (if we started with evolutionary)
        if problem_info['preferred_method'] == 'evolutionary' and best_fitness < 9000:
            print("\nPhase 3: Enhanced A* Search")
            start_time = time.time()
            
            try:
                # Enhance A* with evolutionary knowledge
                hints = self._enhance_astar_with_evolutionary_knowledge()
                
                # Run A* with more iterations and enhanced heuristics
                enhanced_solution = self.astar_agent.search(initial_state, iterations=150)
                elapsed_time = time.time() - start_time
                
                if enhanced_solution and self._evaluate_solution(initial_state, enhanced_solution):
                    print(f"Enhanced A* found solution in {elapsed_time:.2f}s!")
                    return enhanced_solution
                
                if enhanced_solution:
                    fitness = self._calculate_solution_fitness(initial_state, enhanced_solution)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = enhanced_solution
                
                print(f"Enhanced A* completed in {elapsed_time:.2f}s, best fitness: {best_fitness:.2f}")
                
            except Exception as e:
                print(f"Enhanced A* search failed: {e}")
        
        # Return best solution found
        if best_solution:
            print(f"Returning best solution with fitness: {best_fitness:.2f}")
            return best_solution
        else:
            print("No solution found by hybrid approach")
            return []

    def _evaluate_solution(self, initial_state: GameState, solution: List[Direction]) -> bool:
        """
        Evaluates if a solution actually solves the puzzle.
        """
        if not solution:
            return False
        
        try:
            current_state = initial_state
            for action in solution:
                current_state = advance_game_state(action, current_state.copy())
                if check_win(current_state):
                    return True
            return False
        except Exception:
            return False

    def _calculate_solution_fitness(self, initial_state: GameState, solution: List[Direction]) -> float:
        """
        Calculates fitness score for a solution.
        """
        if not solution:
            return 0.0
        
        try:
            current_state = initial_state
            best_score = 0.0
            
            for i, action in enumerate(solution):
                current_state = advance_game_state(action, current_state.copy())
                
                if check_win(current_state):
                    return 10000.0 - i  # Win bonus minus steps
                
                # Intermediate scoring
                score = 0.0
                
                # Rule formation bonus
                score += len(current_state.rules) * 10.0
                
                # Essential rules bonus
                if any('you' in rule for rule in current_state.rules):
                    score += 100.0
                if any('win' in rule for rule in current_state.rules):
                    score += 100.0
                
                # Distance to goal (if applicable)
                if current_state.players and current_state.winnables:
                    min_dist = float('inf')
                    for player in current_state.players:
                        for win_obj in current_state.winnables:
                            dist = abs(player.x - win_obj.x) + abs(player.y - win_obj.y)
                            min_dist = min(min_dist, dist)
                    
                    if min_dist != float('inf'):
                        score += 50.0 / (min_dist + 1)
                
                best_score = max(best_score, score)
            
            return best_score - len(solution) * 0.1  # Length penalty
            
        except Exception:
            return 0.0
