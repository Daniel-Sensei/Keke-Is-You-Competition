import random
from collections import defaultdict
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Any, Dict, Set, FrozenSet
from tqdm import trange
import numpy as np

#
# VERSIONE 16.1 - CORREZIONE BUG KeyError
#
# Corregge un KeyError che si verificava perché la chiave 'behavior'
# non veniva aggiunta al dizionario dell'individuo, causando un crash
# durante l'aggiornamento dell'archivio della novità.
#
class EVOLUTIONARYAgent(BaseAgent):
    def __init__(self, population_size=40, generations=100, mutation_rate=0.15, solution_length=100,
                 w_objective=0.2,
                 w_novelty=0.8,
                 novelty_k=10,
                 novelty_threshold=0.8,
                 rule_dist_weight=10.0):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.solution_length = solution_length
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        self.cache: Dict[Tuple[Direction, ...], GameState] = {}
        
        self.w_objective = w_objective
        self.w_novelty = w_novelty
        self.novelty_k = novelty_k
        self.novelty_threshold = novelty_threshold
        self.rule_dist_weight = rule_dist_weight
        
        self.archive: List[Tuple[Tuple[int, int], FrozenSet[str]]] = []

        self.w_distance = 0.5
        self.w_rule_change = 500.0
        self.w_exploration = 10.0
        self.visited_coords: Set[Tuple[int, int]] = set()

    def _get_manhattan_distance(self, state: GameState) -> float:
        if not state.players or not state.winnables: return 1000.0
        player_pos = [(p.x, p.y) for p in state.players]
        goal_pos = [(g.x, g.y) for g in state.winnables]
        return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)

    def _get_rules(self, state: GameState) -> FrozenSet[str]:
        return frozenset(state.rules)

    def _get_final_state(self, initial_state: GameState, sequence: Tuple[Direction, ...]) -> GameState:
        if sequence in self.cache: return self.cache[sequence]
        current_state = initial_state
        i = 0
        if len(sequence) > 0:
            for i in range(len(sequence), 0, -1):
                prefix = sequence[:i]
                if prefix in self.cache:
                    current_state = self.cache[prefix]
                    break
            else: i = 0
        
        for j in range(i, len(sequence)):
            current_state = advance_game_state(sequence[j], current_state.copy())
            self.cache[sequence[:j+1]] = current_state
        return current_state

    def _get_behavior_characterization(self, state: GameState) -> Tuple[Tuple[int, int], FrozenSet[str]]:
        player_pos = (state.players[0].x, state.players[0].y) if state.players else (-1, -1)
        rules = self._get_rules(state)
        return (player_pos, rules)

    def _get_behavior_distance(self, beh1: Tuple, beh2: Tuple) -> float:
        pos1, rules1 = beh1
        pos2, rules2 = beh2
        pos_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        intersection_size = len(rules1.intersection(rules2))
        union_size = len(rules1.union(rules2))
        rule_dist = 1.0 - (intersection_size / union_size) if union_size > 0 else 0.0
        return pos_dist + self.rule_dist_weight * rule_dist

    def _calculate_novelty(self, behavior: Tuple) -> float:
        if not self.archive:
            return self.novelty_threshold * 2
        
        distances = [self._get_behavior_distance(behavior, other) for other in self.archive]
        distances.sort()
        k_neighbors = distances[:self.novelty_k]
        return np.mean(k_neighbors) if k_neighbors else 0.0

    def _calculate_objective_fitness(self, initial_state: GameState, sequence: List[Direction]) -> Tuple[float, int, GameState]:
        best_intermediate_fitness = -float('inf')
        current_state = initial_state
        previous_rules = self._get_rules(initial_state)

        for k, action in enumerate(sequence):
            prefix = tuple(sequence[:k+1])
            current_state = self._get_final_state(initial_state, prefix)
            
            if check_win(current_state):
                win_step = k + 1
                fitness = 10000.0 - (win_step * 10)
                return fitness, win_step, current_state

            rule_change_bonus = 0.0
            current_rules = self._get_rules(current_state)
            if current_rules != previous_rules:
                rule_change_bonus = self.w_rule_change
                previous_rules = current_rules

            exploration_bonus = 0.0
            if current_state.players:
                player_coord = (current_state.players[0].x, current_state.players[0].y)
                if player_coord not in self.visited_coords:
                    exploration_bonus = self.w_exploration
                    self.visited_coords.add(player_coord)
            
            distance_fitness = -1000.0 if not current_state.players else 100.0 / (self._get_manhattan_distance(current_state) + 1)
            current_fitness = (distance_fitness * self.w_distance) + rule_change_bonus + exploration_bonus
            
            if current_fitness > best_intermediate_fitness:
                best_intermediate_fitness = current_fitness

        final_state = self._get_final_state(initial_state, tuple(sequence))
        return best_intermediate_fitness - (len(sequence) * 0.01), -1, final_state

    def _simplify_solution(self, initial_state: GameState, sequence: List[Direction]) -> List[Direction]:
        simplified_sequence = list(sequence)
        i = 0
        while i < len(simplified_sequence):
            test_sequence = simplified_sequence[:i] + simplified_sequence[i+1:]
            if not test_sequence: break
            final_state = self._get_final_state(initial_state, tuple(test_sequence))
            if check_win(final_state):
                simplified_sequence = test_sequence
            else:
                i += 1
        return simplified_sequence

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        self.cache.clear()
        self.visited_coords.clear()
        self.archive = []
        population = [[random.choice(self.possible_actions) for _ in range(self.solution_length)] for _ in range(self.population_size)]

        t = trange(self.generations, desc="Direct Evolution (v16.1 - Bugfix)")
        for gen in t:
            pop_with_fitness = []

            for seq in population:
                objective_fitness, win_step, final_state = self._calculate_objective_fitness(initial_state, seq)
                
                if win_step != -1:
                    print(f"\nSoluzione trovata alla generazione {gen}! Semplificazione in corso...")
                    return self._simplify_solution(initial_state, seq[:win_step])
                
                behavior = self._get_behavior_characterization(final_state)
                novelty_score = self._calculate_novelty(behavior)
                
                combined_fitness = (self.w_objective * objective_fitness) + (self.w_novelty * novelty_score)
                
                # --- INIZIO DELLA CORREZIONE ---
                pop_with_fitness.append({
                    'seq': seq,
                    'fitness': combined_fitness,
                    'rules': behavior[1],
                    'novelty': novelty_score,
                    'behavior': behavior  # <-- LA CHIAVE MANCANTE È STATA AGGIUNTA QUI
                })
                # --- FINE DELLA CORREZIONE ---

            for ind in pop_with_fitness:
                if ind['novelty'] > self.novelty_threshold:
                    if ind['behavior'] not in self.archive:
                         self.archive.append(ind['behavior'])

            species = defaultdict(list)
            for individual in pop_with_fitness:
                species[individual['rules']].append(individual)
            
            pop_with_shared_fitness = []
            for rule_set, individuals in species.items():
                niche_size = len(individuals)
                for ind in individuals:
                    shared_fitness = ind['fitness'] / niche_size
                    pop_with_shared_fitness.append({'seq': ind['seq'], 'shared_fitness': shared_fitness})
            
            pop_with_shared_fitness.sort(key=lambda x: x['shared_fitness'], reverse=True)
            
            best_fitness_in_gen = pop_with_shared_fitness[0]['shared_fitness'] if pop_with_shared_fitness else 0.0
            t.set_postfix_str(f"Best Fitness: {best_fitness_in_gen:.2f}, Niches: {len(species)}, Archive Size: {len(self.archive)}")

            elites_count = int(self.population_size * 0.1)
            next_generation = [d['seq'] for d in pop_with_shared_fitness[:elites_count]]
            
            sequences_for_selection = [d['seq'] for d in pop_with_shared_fitness]
            selection_weights = [d['shared_fitness'] for d in pop_with_shared_fitness]
            min_fitness = min(selection_weights) if selection_weights else 0
            selection_weights = [(s - min_fitness) + 1e-6 for s in selection_weights]

            while len(next_generation) < self.population_size:
                if sum(selection_weights) > 0:
                    parents = random.choices(sequences_for_selection, weights=selection_weights, k=2)
                else:
                    parents = random.sample(sequences_for_selection, k=2) if len(sequences_for_selection) >= 2 else [random.choice(sequences_for_selection), random.choice(sequences_for_selection)]

                point = random.randint(1, self.solution_length - 1)
                child = parents[0][:point] + parents[1][point:]
                for i in range(len(child)):
                    if random.random() < self.mutation_rate: child[i] = random.choice(self.possible_actions)
                next_generation.append(child)
                
            population = next_generation

        t.close()
        print("No solution found, returning best sequence from the final population.")
        if not pop_with_shared_fitness:
             return [] # Restituisce una soluzione vuota se non trova nulla
        return pop_with_shared_fitness[0]['seq']