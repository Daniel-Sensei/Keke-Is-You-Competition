"""
Final Improved Hybrid Agent for KekeAI Game.

This agent addresses the key issues:
1. Less aggressive move optimization to preserve valid solutions
2. Better validation and solution handling
3. Multiple search strategies without complex classification
"""

import random
import time
import os
import importlib.util
from typing import List, Dict, Tuple, Any
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win


class FINALHYBRIDAgent(BaseAgent):
    """
    Final hybrid agent that combines multiple approaches with careful solution validation.
    """
    
    def __init__(self, max_time_total=60.0):
        super().__init__()
        self.max_time_total = max_time_total
        
        # Initialize multiple agent configurations
        self.agents = self._initialize_agents()

    def _initialize_agents(self):
        """Initialize multiple agent configurations for different strategies."""
        agents = {}
        
        try:
            # A* agent
            astar_class = self._load_agent_class('improved_astar_AGENT.py', 'IMPROVED_ASTARAgent')
            agents['astar'] = astar_class()
            
            # Conservative evolutionary agent (less optimization)
            evo_class = self._load_agent_class('improved_evolutionary_AGENT.py', 'IMPROVED_EVOLUTIONARYAgent')
            agents['evo_conservative'] = evo_class(
                population_size=30, 
                generations=40, 
                solution_length=50
            )
            
            # Aggressive evolutionary agent (more exploration)
            agents['evo_aggressive'] = evo_class(
                population_size=60, 
                generations=80, 
                solution_length=80
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize agents: {e}")
            
        return agents

    def _load_agent_class(self, agent_filename, class_name):
        """Dynamically loads an agent class from a file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            agent_path = os.path.join(current_dir, agent_filename)
            
            spec = importlib.util.spec_from_file_location(class_name, agent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            agent_class = getattr(module, class_name)
            return agent_class
            
        except Exception as e:
            print(f"[ERROR] Failed to load agent {class_name} from {agent_filename}: {e}")
            return BaseAgent

    def _validate_solution_carefully(self, initial_state: GameState, solution: List[Direction]) -> Tuple[bool, int]:
        """
        Carefully validates solution and returns (is_valid, winning_step).
        """
        if not solution:
            return False, -1
        
        try:
            current_state = initial_state
            for i, action in enumerate(solution):
                current_state = advance_game_state(action, current_state.copy())
                if check_win(current_state):
                    return True, i + 1
            
            return False, -1
            
        except Exception as e:
            print(f"[DEBUG] Solution validation error: {e}")
            return False, -1

    def _truncate_at_win(self, initial_state: GameState, solution: List[Direction]) -> List[Direction]:
        """Truncates solution at the first winning state."""
        is_valid, win_step = self._validate_solution_carefully(initial_state, solution)
        
        if is_valid and win_step > 0:
            truncated = solution[:win_step]
            print(f"[DEBUG] Solution truncated from {len(solution)} to {len(truncated)} moves")
            return truncated
        
        return solution

    def _force_simple_strategy(self, agent):
        """Forces an evolutionary agent to use simple movement strategy."""
        if hasattr(agent, '_classify_level_type'):
            original_classify = agent._classify_level_type
            agent._classify_level_type = lambda x: "simple_movement"
            return original_classify
        return None

    def _restore_classify(self, agent, original_classify):
        """Restores original classification function."""
        if original_classify and hasattr(agent, '_classify_level_type'):
            agent._classify_level_type = original_classify

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        """
        Multi-phase hybrid search with careful solution validation.
        """
        print("Starting final hybrid search...")
        start_time = time.time()
        
        valid_solutions = []
        all_attempts = []
        
        # Phase 1: Quick A* (good for simple rule formation)
        print("Phase 1: Quick A* Search")
        if 'astar' in self.agents:
            try:
                astar_solution = self.agents['astar'].search(initial_state, iterations=30)
                elapsed = time.time() - start_time
                
                if astar_solution:
                    is_valid, win_step = self._validate_solution_carefully(initial_state, astar_solution)
                    if is_valid:
                        final_solution = self._truncate_at_win(initial_state, astar_solution)
                        print(f"A* found valid solution in {elapsed:.2f}s: {len(final_solution)} moves")
                        return final_solution
                    else:
                        all_attempts.append(('A* Quick', astar_solution, elapsed))
                        print(f"A* found invalid solution: {len(astar_solution)} moves")
                else:
                    print(f"A* found no solution in {elapsed:.2f}s")
                    
            except Exception as e:
                print(f"A* search failed: {e}")

        # Phase 2: Conservative Evolutionary (balanced approach)
        print("Phase 2: Conservative Evolutionary Search")
        if 'evo_conservative' in self.agents and time.time() - start_time < self.max_time_total * 0.6:
            try:
                agent = self.agents['evo_conservative']
                original_classify = self._force_simple_strategy(agent)
                
                evo_solution = agent.search(initial_state, iterations)
                elapsed = time.time() - start_time
                
                self._restore_classify(agent, original_classify)
                
                if evo_solution:
                    is_valid, win_step = self._validate_solution_carefully(initial_state, evo_solution)
                    if is_valid:
                        final_solution = self._truncate_at_win(initial_state, evo_solution)
                        print(f"Conservative Evolutionary found valid solution in {elapsed:.2f}s: {len(final_solution)} moves")
                        return final_solution
                    else:
                        all_attempts.append(('Conservative Evo', evo_solution, elapsed))
                        print(f"Conservative Evolutionary found invalid solution: {len(evo_solution)} moves")
                else:
                    print(f"Conservative Evolutionary found no solution in {elapsed:.2f}s")
                    
            except Exception as e:
                print(f"Conservative Evolutionary search failed: {e}")

        # Phase 3: Extended A* (for levels that need more iterations)
        print("Phase 3: Extended A* Search")
        if 'astar' in self.agents and time.time() - start_time < self.max_time_total * 0.8:
            try:
                astar_ext_solution = self.agents['astar'].search(initial_state, iterations=100)
                elapsed = time.time() - start_time
                
                if astar_ext_solution:
                    is_valid, win_step = self._validate_solution_carefully(initial_state, astar_ext_solution)
                    if is_valid:
                        final_solution = self._truncate_at_win(initial_state, astar_ext_solution)
                        print(f"Extended A* found valid solution in {elapsed:.2f}s: {len(final_solution)} moves")
                        return final_solution
                    else:
                        all_attempts.append(('A* Extended', astar_ext_solution, elapsed))
                        print(f"Extended A* found invalid solution: {len(astar_ext_solution)} moves")
                else:
                    print(f"Extended A* found no solution in {elapsed:.2f}s")
                    
            except Exception as e:
                print(f"Extended A* search failed: {e}")

        # Phase 4: Aggressive Evolutionary (if time permits)
        print("Phase 4: Aggressive Evolutionary Search")
        if 'evo_aggressive' in self.agents and time.time() - start_time < self.max_time_total * 0.95:
            try:
                agent = self.agents['evo_aggressive']
                original_classify = self._force_simple_strategy(agent)
                
                evo_agg_solution = agent.search(initial_state, iterations)
                elapsed = time.time() - start_time
                
                self._restore_classify(agent, original_classify)
                
                if evo_agg_solution:
                    is_valid, win_step = self._validate_solution_carefully(initial_state, evo_agg_solution)
                    if is_valid:
                        final_solution = self._truncate_at_win(initial_state, evo_agg_solution)
                        print(f"Aggressive Evolutionary found valid solution in {elapsed:.2f}s: {len(final_solution)} moves")
                        return final_solution
                    else:
                        all_attempts.append(('Aggressive Evo', evo_agg_solution, elapsed))
                        print(f"Aggressive Evolutionary found invalid solution: {len(evo_agg_solution)} moves")
                else:
                    print(f"Aggressive Evolutionary found no solution in {elapsed:.2f}s")
                    
            except Exception as e:
                print(f"Aggressive Evolutionary search failed: {e}")

        # Return best invalid solution if no valid one found
        if all_attempts:
            print(f"No valid solution found, selecting best from {len(all_attempts)} attempts")
            
            # Sort by solution length (prefer shorter)
            all_attempts.sort(key=lambda x: len(x[1]) if x[1] else float('inf'))
            
            best_method, best_solution, best_time = all_attempts[0]
            print(f"Returning best attempt from {best_method}: {len(best_solution)} moves")
            
            # Try to at least truncate at any improvement
            return self._truncate_at_win(initial_state, best_solution)
        
        print("No solution found by any method")
        return []
