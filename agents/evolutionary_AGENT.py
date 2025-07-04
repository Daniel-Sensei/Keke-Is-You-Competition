import random
from collections import defaultdict
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Dict, Set
from tqdm import trange
import numpy as np

class EVOLUTIONARYAgent(BaseAgent):
    """
    VERSIONE 2.2 - MAP-ELITES PER LA DIVERSITÀ

    Questa versione introduce un meccanismo ispirato a MAP-Elites per mantenere
    la diversità genetica. Invece di un semplice elitismo, l'algoritmo
    mantiene un archivio dei migliori individui per ogni "nicchia" di comportamento,
    selezionando i genitori da questo archivio diversificato per evitare
    la convergenza prematura e risolvere puzzle più complessi.
    """
    def __init__(self, population_size=50, generations=100, mutation_rate=0.2, solution_length=100,
                 w_objective=0.3, w_novelty=0.7, novelty_k=15, novelty_threshold=0.8,
                 rule_dist_weight=10.0, archive_size_limit=500):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.solution_length = solution_length
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        
        self.w_objective = w_objective
        self.w_novelty = w_novelty
        self.w_distance = 0.5
        self.w_rule_change = 500.0
        self.w_exploration = 10.0
        
        self.novelty_k = novelty_k
        self.novelty_threshold = novelty_threshold
        self.rule_dist_weight = rule_dist_weight
        self.archive_size_limit = archive_size_limit
        self.archive: List[Tuple[Tuple[int, int], frozenset]] = []
        
        # NUOVO: Mappa degli individui elite per comportamento
        self.elite_map: Dict[Tuple, Dict] = {}
        
        self.visited_coords: Set[Tuple[int, int]] = set()

    def _get_manhattan_distance(self, state: GameState) -> float:
        if not state.players or not state.winnables: return 1000.0
        player_pos = [(p.x, p.y) for p in state.players]
        goal_pos = [(g.x, g.y) for g in state.winnables]
        if not player_pos or not goal_pos: return 1000.0
        return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)

    def _get_rules(self, state: GameState) -> frozenset:
        return frozenset(state.rules)

    def _get_behavior_characterization(self, state: GameState) -> Tuple[Tuple[int, int], frozenset]:
        player_pos = (state.players[0].x, state.players[0].y) if state.players else (-1, -1)
        rules = self._get_rules(state)
        return (player_pos, rules)

    def _get_behavior_distance(self, beh1: Tuple, beh2: Tuple) -> float:
        pos1, rules1 = beh1
        pos2, rules2 = beh2
        pos_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        union_size = len(rules1.union(rules2))
        rule_dist = 1.0 - (len(rules1.intersection(rules2)) / union_size) if union_size > 0 else 0.0
        return pos_dist + self.rule_dist_weight * rule_dist

    def _calculate_novelty(self, behavior: Tuple) -> float:
        if not self.archive: return self.novelty_threshold * 2
        distances = [self._get_behavior_distance(behavior, other) for other in self.archive]
        distances.sort()
        return np.mean(distances[:self.novelty_k]) if len(distances) > self.novelty_k else np.mean(distances)

    def _evaluate_individual(self, initial_state: GameState, sequence: List[Direction]) -> Dict:
        current_state = initial_state
        previous_rules = self._get_rules(initial_state)
        min_distance = self._get_manhattan_distance(initial_state)
        total_rule_changes, total_exploration_bonus = 0, 0.0
        
        self.visited_coords.clear()
        if current_state.players: self.visited_coords.add((current_state.players[0].x, current_state.players[0].y))

        for k, action in enumerate(sequence):
            current_state = advance_game_state(action, current_state.copy())
            if check_win(current_state):
                return {"win": True, "win_step": k + 1}
            
            min_distance = min(min_distance, self._get_manhattan_distance(current_state))
            current_rules = self._get_rules(current_state)
            if current_rules != previous_rules:
                total_rule_changes += 1
                previous_rules = current_rules
            if current_state.players:
                player_coord = (current_state.players[0].x, current_state.players[0].y)
                if player_coord not in self.visited_coords:
                    total_exploration_bonus += self.w_exploration
                    self.visited_coords.add(player_coord)
                    
        distance_fitness = 100.0 / (min_distance + 1)
        objective_fitness = (distance_fitness * self.w_distance) + (total_rule_changes * self.w_rule_change) + total_exploration_bonus
        
        return {"win": False, "objective_fitness": objective_fitness, "behavior": self._get_behavior_characterization(current_state)}

    def _simplify_solution(self, initial_state: GameState, sequence: List[Direction]) -> List[Direction]:
        simplified = list(sequence)
        i = 0
        while i < len(simplified):
            test_sequence = simplified[:i] + simplified[i+1:]
            if not test_sequence: break
            final_state = initial_state
            for action in test_sequence: final_state = advance_game_state(action, final_state.copy())
            if check_win(final_state): simplified = test_sequence
            else: i += 1
        return simplified

    def _create_initial_population(self, initial_state: GameState) -> List[List[Direction]]:
        population = []
        num_greedy = self.population_size // 2
        for _ in range(num_greedy):
            sequence, current_state = [], initial_state.copy()
            initial_goal_pos = (initial_state.winnables[0].x, initial_state.winnables[0].y) if initial_state.winnables else None
            for _ in range(self.solution_length):
                best_action = random.choice(self.possible_actions)
                if current_state.players and initial_goal_pos:
                    min_dist = float('inf')
                    for action in self.possible_actions:
                        next_s = advance_game_state(action, current_state.copy())
                        if next_s.players:
                            dist = abs(next_s.players[0].x - initial_goal_pos[0]) + abs(next_s.players[0].y - initial_goal_pos[1])
                            if dist < min_dist: min_dist, best_action = dist, action
                sequence.append(best_action)
                current_state = advance_game_state(best_action, current_state.copy())
            population.append(sequence)
        for _ in range(self.population_size - num_greedy):
            population.append([random.choice(self.possible_actions) for _ in range(self.solution_length)])
        return population

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        self.archive, self.elite_map = [], {}
        population = self._create_initial_population(initial_state)

        t = trange(self.generations, desc="Direct Evolution (v2.2 - MAP-Elites)")
        for gen in t:
            individuals = []
            for seq in population:
                eval_results = self._evaluate_individual(initial_state, seq)
                if eval_results["win"]:
                    print(f"\nSoluzione trovata alla generazione {gen}! Semplificazione...")
                    return self._simplify_solution(initial_state, seq[:eval_results["win_step"]])
                
                novelty_score = self._calculate_novelty(eval_results["behavior"])
                fitness = (self.w_objective * eval_results["objective_fitness"]) + (self.w_novelty * novelty_score)
                individuals.append({'seq': seq, 'fitness': fitness, 'behavior': eval_results["behavior"], 'novelty': novelty_score})

            # Aggiorna la mappa degli elite e l'archivio della novità
            for ind in individuals:
                behavior = ind['behavior']
                if ind['novelty'] > self.novelty_threshold and behavior not in self.archive:
                    self.archive.append(behavior)
                    if len(self.archive) > self.archive_size_limit: self.archive.pop(random.randrange(len(self.archive)))
                
                # Se la nicchia è vuota o l'individuo è migliore del campione attuale, lo si salva
                if behavior not in self.elite_map or ind['fitness'] > self.elite_map[behavior]['fitness']:
                    self.elite_map[behavior] = ind

            # Riproduzione: crea la nuova popolazione dagli elite
            elite_parents = list(self.elite_map.values())
            if not elite_parents: # Failsafe se la mappa è vuota
                elite_parents = individuals or [{'seq': p} for p in population]

            next_generation = []
            while len(next_generation) < self.population_size:
                # Scegli due genitori unici dall'elite map
                p1, p2 = random.sample(elite_parents, 2) if len(elite_parents) > 1 else (random.choice(elite_parents), random.choice(elite_parents))
                
                point = random.randint(1, self.solution_length - 1)
                child = p1['seq'][:point] + p2['seq'][point:]
                for i in range(len(child)):
                    if random.random() < self.mutation_rate: child[i] = random.choice(self.possible_actions)
                next_generation.append(child)
            
            population = next_generation
            
            best_fitness_gen = max(ind['fitness'] for ind in individuals) if individuals else 0
            t.set_postfix_str(f"Best Fitness: {best_fitness_gen:.2f}, Elite Niches: {len(self.elite_map)}, Archive: {len(self.archive)}")
            
        t.close()
        print("Nessuna soluzione trovata.")
        return []