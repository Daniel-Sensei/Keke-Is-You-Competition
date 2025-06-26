"""
Evolutionary Algorithm (EA) Agent for KekeAI.

This agent uses an evolutionary approach to find solutions for Baba Is You style puzzles.
A population of action sequences (chromosomes) is evolved over generations.
Each sequence's fitness is evaluated based on how well it solves or progresses in the puzzle.
Genetic operators (selection, crossover, mutation) are used to create new generations.

Key Features:
- Fitness function includes rewards for winning, forming new rules, and reducing
  distance to winnable objects. Penalties for player elimination, loops, and
  exceeding evaluation steps.
- Tuple-based state hashing for efficient loop detection within sequence evaluations.
- Standard genetic operators: tournament selection, single-point crossover, random mutation.
- Elitism to preserve best individuals.
- Parameters (population size, sequence length, mutation/crossover rates) are tunable.
"""
import random
from typing import List, Tuple, Optional

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, double_map_to_string # double_map_to_string might still be needed if _get_state_hash is removed/changed later

# --- EA Parameters ---
# These parameters control the behavior of the evolutionary algorithm.
# They may require tuning for optimal performance on different level sets.

POPULATION_SIZE = 30         # Number of individuals (action sequences) in the population.
                             # Larger populations explore more but are slower per generation.

MAX_SEQUENCE_LENGTH = 40     # Maximum length of an action sequence.
                             # Should be long enough for solutions, but not excessively long to keep search space manageable.

MUTATION_RATE = 0.05         # Probability that each action (gene) in a sequence will be mutated.
                             # Helps maintain diversity and explore new solutions.

CROSSOVER_RATE = 0.8         # Probability that crossover will occur between two selected parents.
                             # If no crossover, parents are cloned.

ELITISM_COUNT = 2            # Number of the best individuals from the current generation that are
                             # directly carried over to the next generation, unchanged. Ensures best solutions are not lost.

TOURNAMENT_SIZE = 3          # Size of the tournament used in tournament selection.
                             # Larger tournaments increase selection pressure.

SOLUTION_LENGTH_PENALTY_FACTOR = 0.001 # Factor to penalize solutions by their length.
                                     # Encourages finding shorter solutions when fitness scores are otherwise similar.

# Maximum number of game steps to simulate for a single individual's fitness evaluation.
# This prevents fitness calculation from getting stuck in very long, non-productive (or looping) sequences.
MAX_EVALUATION_STEPS = MAX_SEQUENCE_LENGTH + 10 # Allow some buffer beyond the max sequence length.


# Class name must match the filename prefix, capitalized, for execution.py to load it.
class EVOLUTIONARYAgent(BaseAgent): 
    """
    Implements an Evolutionary Algorithm (EA) to find solutions for KekeAI puzzles.

    The agent evolves a population of action sequences over a number of generations.
    Fitness of each sequence is determined by simulating its effect on the game state,
    rewarding sequences that solve the puzzle or make significant progress.
    """

    def __init__(self):
        super().__init__()
        # EA parameters are currently module-level constants for simplicity.
        # They could be made instance variables if different configurations per agent instance were needed.

    def _get_state_hash(self, game_state: GameState) -> tuple:
        """
        Creates a hashable tuple representing the core components of the game state.
        This is used for detecting visited states to avoid loops during sequence evaluation.
        Includes positions of physical objects, word objects, and the set of active rules.
        Objects are sorted to ensure a consistent hash for the same semantic state.
        """
        phys_tuples = []
        # GameState.phys can sometimes contain non-GameObj elements (like ' ') if not cleaned up, filter them.
        if game_state.phys:
            for p in sorted([obj for obj in game_state.phys if hasattr(obj, 'name')], key=lambda obj: (obj.name, obj.x, obj.y)):
                phys_tuples.append((p.name, p.x, p.y))
        
        word_tuples = []
        all_word_like_objects = []
        if game_state.words: # List of GameObj
            all_word_like_objects.extend([obj for obj in game_state.words if hasattr(obj, 'name')])
        if game_state.keywords: # List of GameObj
            all_word_like_objects.extend([obj for obj in game_state.keywords if hasattr(obj, 'name')])
            
        if all_word_like_objects:
            for w in sorted(all_word_like_objects, key=lambda obj: (obj.name, obj.x, obj.y)):
                word_tuples.append((w.name, w.x, w.y))
        
        # Ensure rules are sorted for consistent tuple hashing
        rules_tuple = tuple(sorted(list(set(game_state.rules))))
        
        # Consider adding player positions if not already covered by phys_tuples and distinct
        # player_pos_tuples = []
        # if game_state.players:
        #     for p in sorted([obj for obj in game_state.players if hasattr(obj, 'name')], key=lambda obj: (obj.name, obj.x, obj.y)):
        #         player_pos_tuples.append((p.name, p.x, p.y))
        # return (tuple(phys_tuples), tuple(word_tuples), rules_tuple, tuple(player_pos_tuples))

        return (tuple(phys_tuples), tuple(word_tuples), rules_tuple)

    def _initialize_individual(self) -> List[Direction]:
        """
        Creates a single random individual (action sequence).
        The sequence is padded with Direction.Wait if shorter than MAX_SEQUENCE_LENGTH,
        though initially, we'll just create sequences of fixed MAX_SEQUENCE_LENGTH.
        """
        return [random.choice(list(Direction)[:-1]) for _ in range(MAX_SEQUENCE_LENGTH)] # Exclude Undefined

    def _initialize_population(self) -> List[List[Direction]]:
        """
        Creates an initial population of random individuals.
        """
        return [self._initialize_individual() for _ in range(POPULATION_SIZE)]

    def _calculate_fitness(self, sequence: List[Direction], initial_state: GameState) -> float:
        """
        Calculates the fitness of an individual (action sequence).
        Applies the sequence to a copy of the initial_state and evaluates the outcome.

        Args:
            sequence: The list of Direction enums representing the actions.
            initial_state: The starting GameState of the puzzle.

        Returns:
            A float representing the fitness score. Higher is better.
        """
        current_state = initial_state.copy()
        
        visited_states_in_eval = {self._get_state_hash(current_state)} # Use new hash function
        
        fitness = 0.0
        # Heuristic reward/penalty values
        WIN_BONUS = 1000.0
        PLAYER_ELIMINATED_PENALTY = -200.0
        LOOP_PENALTY = -150.0
        MAX_EVAL_STEPS_PENALTY = -100.0
        
        RULE_FORMATION_BONUS = 10.0
        DISTANCE_REDUCTION_MAX_BONUS = 20.0

        initial_player_positions = {} 
        initial_winnable_positions = {}

        if current_state.players:
            for p in current_state.players:
                player_key = p.name 
                if player_key not in initial_player_positions:
                    initial_player_positions[player_key] = (p.x, p.y)
        
        if current_state.winnables:
            for w in current_state.winnables:
                winnable_key = w.name
                if winnable_key not in initial_winnable_positions:
                     initial_winnable_positions[winnable_key] = (w.x, w.y)

        initial_rules_set = set(current_state.rules)

        for step_idx, action in enumerate(sequence):
            if step_idx >= MAX_EVALUATION_STEPS: # MAX_EVALUATION_STEPS is based on current MAX_SEQUENCE_LENGTH
                fitness += MAX_EVAL_STEPS_PENALTY
                break 

            current_state = advance_game_state(action, current_state)
            current_state_hash = self._get_state_hash(current_state) # Use new hash function

            if current_state_hash in visited_states_in_eval:
                fitness += LOOP_PENALTY 
                break 
            visited_states_in_eval.add(current_state_hash)

            if not current_state.players:
                fitness += PLAYER_ELIMINATED_PENALTY 
                break 
            
            if check_win(current_state):
                fitness += WIN_BONUS
                fitness -= (step_idx + 1) * SOLUTION_LENGTH_PENALTY_FACTOR 
                return fitness, step_idx + 1

            current_rules_set = set(current_state.rules)
            newly_formed_rules_count = len(current_rules_set - initial_rules_set)
            if newly_formed_rules_count > 0:
                 fitness += RULE_FORMATION_BONUS * newly_formed_rules_count
            # Consider if initial_rules_set should be updated here for incremental rule bonus
            # For now, it's compared against the very start of this individual's evaluation.

            if current_state.players and current_state.winnables:
                best_dist_reduction_contribution = 0.0
                for p_key, p_initial_pos in initial_player_positions.items():
                    current_player_obj = next((p for p in current_state.players if p.name == p_key), None)
                    if not current_player_obj: continue

                    for w_key, w_initial_pos in initial_winnable_positions.items():
                        current_winnable_obj = next((w for w in current_state.winnables if w.name == w_key), None)
                        if not current_winnable_obj: continue
                        
                        initial_dist = abs(p_initial_pos[0] - w_initial_pos[0]) + \
                                       abs(p_initial_pos[1] - w_initial_pos[1])
                        current_dist = abs(current_player_obj.x - current_winnable_obj.x) + \
                                       abs(current_player_obj.y - current_winnable_obj.y)
                        
                        if initial_dist > 0:
                            reduction_score = (initial_dist - current_dist) / initial_dist 
                            if reduction_score > 0 :
                                best_dist_reduction_contribution = max(best_dist_reduction_contribution, reduction_score * DISTANCE_REDUCTION_MAX_BONUS)
                fitness += best_dist_reduction_contribution

        fitness -= len(sequence) * SOLUTION_LENGTH_PENALTY_FACTOR
        return fitness, None


    def _selection(self, population: List[List[Direction]], fitnesses: List[float]) -> List[Direction]:
        """
        Selects one individual from the population based on fitness.
        Uses tournament selection.
        """
        """
        Selects one individual (parent) from the population using tournament selection.
        A random subset of individuals (tournament) is chosen from the population,
        and the fittest individual from this subset is selected.

        Args:
            population: The current population of action sequences.
            fitnesses: A list of fitness scores corresponding to the individuals in the population.

        Returns:
            The selected individual (action sequence).
        """
        # Ensure tournament size is not larger than population size
        actual_tournament_size = min(TOURNAMENT_SIZE, len(population))
        if actual_tournament_size == 0: # Should not happen if population is not empty
            return self._initialize_individual() # Fallback, though problematic

        tournament_candidate_indices = random.sample(range(len(population)), actual_tournament_size)
        
        best_candidate_from_tournament_idx = -1
        best_fitness_in_tournament = -float('inf')
        
        for index in tournament_candidate_indices:
            if fitnesses[index] > best_fitness_in_tournament:
                best_fitness_in_tournament = fitnesses[index]
                best_candidate_from_tournament_idx = index
        
        return population[best_candidate_from_tournament_idx]

    def _crossover(self, parent1: List[Direction], parent2: List[Direction]) -> Tuple[List[Direction], List[Direction]]:
        """
        Performs single-point crossover between two parent sequences to produce two offspring sequences.
        Crossover occurs with a probability defined by `CROSSOVER_RATE`.
        If crossover doesn't occur, copies of the parents are returned.

        Args:
            parent1: The first parent sequence.
            parent2: The second parent sequence.

        Returns:
            A tuple containing two offspring sequences.
        """
        offspring1 = parent1[:] # Start with copies
        offspring2 = parent2[:]

        if random.random() < CROSSOVER_RATE and \
           len(parent1) > 1 and len(parent2) > 1: # Ensure crossover is possible
            # Choose a random crossover point (excluding the very start and end)
            # Ensures that both parts of the sequence have at least one element.
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            
        return offspring1, offspring2

    def _mutate(self, sequence: List[Direction]) -> List[Direction]:
        """
        Performs mutation on an individual action sequence.
        Each action (gene) in the sequence has a `MUTATION_RATE` chance of being
        replaced by a new random valid action.

        Args:
            sequence: The action sequence to mutate.

        Returns:
            The mutated action sequence.
        """
        mutated_sequence = sequence[:] # Work on a copy to avoid modifying the original directly
        possible_actions = [d for d in Direction if d != Direction.Undefined] # Cache this list

        for i in range(len(mutated_sequence)):
            if random.random() < MUTATION_RATE:
                mutated_sequence[i] = random.choice(possible_actions)
        return mutated_sequence

    def search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """
        The main evolutionary algorithm loop.
        `iterations` here corresponds to the number of generations.
        """
        # The main evolutionary algorithm loop.
        # `iterations` here corresponds to the number of generations.
        population = self._initialize_population()
        best_solution_overall: List[Direction] = [] 
        best_fitness_overall = -float('inf')
        best_sol_effective_length: Optional[int] = None


        for generation in range(iterations): # iterations is number of generations
            # Each item in population_eval_results is (fitness_score, effective_length_or_None)
            population_eval_results = [self._calculate_fitness(ind, initial_state) for ind in population]
            
            current_fitnesses = [res[0] for res in population_eval_results]

            # Track overall best solution found so far
            for i, (fitness_val, eff_len) in enumerate(population_eval_results):
                if fitness_val > best_fitness_overall:
                    best_fitness_overall = fitness_val
                    best_solution_overall = population[i][:] # Store a copy
                    best_sol_effective_length = eff_len # Store its effective length if it's a win

                # Check for solution (eff_len will be non-None if it's a win)
                if eff_len is not None: 
                    # A winning sequence was found
                    return population[i][:eff_len] # Return trimmed solution

            # Create the next generation
            new_population: List[List[Direction]] = []

            # Elitism: Carry over the best individuals from the current generation
            # Sort population by fitness (descending) to easily pick elites
            sorted_population_for_elitism = sorted(zip(population, current_fitnesses), key=lambda x_item: x_item[1], reverse=True)
            
            for i in range(ELITISM_COUNT):
                if i < len(sorted_population_for_elitism):
                    new_population.append(sorted_population_for_elitism[i][0][:]) # Add copy of elite individual

            # Fill the rest of the new population with offspring
            while len(new_population) < POPULATION_SIZE:
                # Select parents from the current population using their fitnesses
                parent1 = self._selection(population, current_fitnesses) 
                parent2 = self._selection(population, current_fitnesses)
                
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                new_population.append(self._mutate(offspring1))
                if len(new_population) < POPULATION_SIZE: # Ensure not to overfill
                    new_population.append(self._mutate(offspring2))
            
            population = new_population # The new generation becomes the current population for the next iteration
            
            # Optional: Logging for generation progress
            # gen_best_fitness_val = sorted_population_for_elitism[0][1] if sorted_population_for_elitism else -float('inf')
            # print(f"Generation {generation + 1}/{iterations}: Best Fitness in Gen = {gen_best_fitness_val:.2f}, Overall Best Fitness = {best_fitness_overall:.2f}")


        # After all generations, return the best solution found overall
        if best_solution_overall:
            if best_sol_effective_length is not None:
                # If the overall best was a winning sequence, trim to its effective length
                return best_solution_overall[:best_sol_effective_length]
            else:
                # If overall best was not a win, return the full sequence (or trim trailing waits for aesthetics)
                final_solution = best_solution_overall[:]
                while final_solution and final_solution[-1] == Direction.Wait:
                    final_solution.pop()
                return final_solution
        
        return [] # Should only be reached if population was empty or no solution ever found.


if __name__ == '__main__':
    # Placeholder for potential direct testing
    # Needs a way to create/load GameState
    print("EvolutionaryAgent defined. Ready for framework integration and testing.")
    # Example (requires game setup from elsewhere):
    # from baba import make_level, parse_map
    # test_map_ascii = "__________\n_B12..F13_\n_........_\n_.b....f._\n__________"
    # game_map = parse_map(test_map_ascii)
    # initial_gs = make_level(game_map)
    # agent = EvolutionaryAgent()
    # solution = agent.search(initial_gs, iterations=100) # 100 generations
    # print(f"EA Solution: {solution}")
