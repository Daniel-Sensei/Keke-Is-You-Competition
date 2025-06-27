import random
from collections import defaultdict
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Any, Dict, Set, FrozenSet
from tqdm import trange
import copy

#
# VERSIONE 15.0 - EVOLUZIONE CON SPECIAZIONE COMPORTAMENTALE
#
# Questa versione implementa una tecnica avanzata per mantenere la diversità
# e risolvere puzzle multi-stadio che ingannavano le versioni precedenti.
# Sostituisce il meccanismo del "cataclisma" con un approccio più elegante.
#
# 1. SPECIAZIONE (NICHING): La popolazione viene suddivisa in "specie" basate
#    sul loro comportamento (definito come l'insieme di regole che creano).
# 2. FITNESS CONDIVISA: Il fitness di ogni individuo viene diviso per il numero
#    di individui nella sua specie. Questo penalizza le strategie sovraffollate
#    (ottimi locali) e premia le soluzioni uniche e innovative.
# 3. CONSERVAZIONE DI STRATEGIE MULTIPLE: Questo approccio permette all'agente di
#    esplorare e mantenere attive più strategie promettenti contemporaneamente,
#    essenziale per i puzzle che richiedono una sequenza di scoperte logiche.
#
class EVOLUTIONARYAgent(BaseAgent):
    """
    Agente evolutivo con speciazione comportamentale per preservare la diversità.
    """
    def __init__(self, population_size=40, generations=100, mutation_rate=0.15, solution_length=80, local_search_steps=5,
                 w_distance=0.5, w_rule_change=500.0, w_exploration=10.0):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.solution_length = solution_length
        self.local_search_steps = local_search_steps
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        self.cache: Dict[Tuple[Direction, ...], GameState] = {}
        
        self.w_distance = w_distance
        self.w_rule_change = w_rule_change
        self.w_exploration = w_exploration
        self.visited_coords: Set[Tuple[int, int]] = set()

    def _get_manhattan_distance(self, state: GameState) -> float:
        if not state.players or not state.winnables: return 1000.0
        player_pos = [(p.x, p.y) for p in state.players]
        goal_pos = [(g.x, g.y) for g in state.winnables]
        return min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)

    def _get_rules(self, state: GameState) -> FrozenSet[str]:
        return frozenset(state.rules)

    def _get_final_state(self, initial_state: GameState, sequence: Tuple[Direction, ...]) -> GameState:
        """Ottiene lo stato finale di una sequenza, usando la cache."""
        if sequence in self.cache:
            return self.cache[sequence]

        current_state = initial_state
        # Ottimizzazione: trova il prefisso più lungo in cache
        for i in range(len(sequence), 0, -1):
            prefix = sequence[:i]
            if prefix in self.cache:
                current_state = self.cache[prefix]
                break
        else: i = 0
        
        # Simula solo la parte non in cache
        for j in range(i, len(sequence)):
            current_state = advance_game_state(sequence[j], current_state.copy())
            self.cache[sequence[:j+1]] = current_state
        
        return current_state
        
    def _calculate_fitness(self, initial_state: GameState, sequence: List[Direction]) -> Tuple[float, int, FrozenSet[str]]:
        """Calcola il fitness e restituisce anche il set di regole finale per la speciazione."""
        best_intermediate_fitness = -float('inf')
        current_state = initial_state
        previous_rules = self._get_rules(initial_state)

        for k, action in enumerate(sequence):
            prefix = tuple(sequence[:k+1])
            current_state = self._get_final_state(initial_state, prefix) # Usa l'helper per la simulazione
            
            if check_win(current_state):
                win_step = k + 1
                return 10000.0 - (win_step * 10), win_step, self._get_rules(current_state)

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

        final_rules = self._get_rules(self._get_final_state(initial_state, tuple(sequence)))
        return best_intermediate_fitness - (len(sequence) * 0.01), -1, final_rules

    def _local_search_enhanced(self, initial_state: GameState, individual: List[Direction]) -> List[Direction]:
        # Questa funzione ora usa _calculate_fitness che restituisce 3 valori, quindi adattiamo l'unpacking
        current_best_seq = list(individual)
        current_best_fitness, _, __ = self._calculate_fitness(initial_state, current_best_seq)

        for _ in range(self.local_search_steps):
            temp_seq = list(current_best_seq)
            # ... (la logica interna rimane la stessa)
            new_fitness, _, __ = self._calculate_fitness(initial_state, temp_seq)
            if new_fitness > current_best_fitness:
                current_best_fitness = new_fitness
                current_best_seq = temp_seq
        return current_best_seq
    
    def _simplify_solution(self, initial_state: GameState, sequence: List[Direction]) -> List[Direction]:
        """
        Tenta di semplificare una sequenza vincente rimuovendo le mosse non necessarie.
        """
        simplified_sequence = list(sequence)
        i = 0
        while i < len(simplified_sequence):
            # Prova a rimuovere la mossa all'indice i
            test_sequence = simplified_sequence[:i] + simplified_sequence[i+1:]
            
            # Simula per vedere se la sequenza più corta vince ancora
            final_state = self._get_final_state(initial_state, tuple(test_sequence))
            
            if check_win(final_state):
                # La rimozione ha avuto successo, la sequenza è ora più corta.
                # Riman_i allo stesso indice per controllare la nuova mossa in questa posizione.
                simplified_sequence = test_sequence
            else:
                # La mossa era necessaria, passa alla successiva.
                i += 1
            
        return simplified_sequence

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        self.cache.clear()
        self.visited_coords.clear()
        population = [[random.choice(self.possible_actions) for _ in range(self.solution_length)] for _ in range(self.population_size)]

        t = trange(self.generations, desc="Direct Evolution (v15 - Speciation)")
        for gen in t:
            # 1. Calcola il fitness GREZZO per tutta la popolazione
            pop_with_fitness = []
            for seq in population:
                raw_fitness, win_step, final_rules = self._calculate_fitness(initial_state, seq)
                if win_step != -1:
                    return self._simplify_solution(initial_state, seq[:win_step])
                pop_with_fitness.append({'seq': seq, 'fitness': raw_fitness, 'rules': final_rules})
            
            # 2. Speciazione: Raggruppa per comportamento (set di regole finali)
            species = defaultdict(list)
            for individual in pop_with_fitness:
                species[individual['rules']].append(individual)
            
            # 3. Calcola il Fitness Condiviso e ordina
            pop_with_shared_fitness = []
            for rule_set, individuals in species.items():
                niche_size = len(individuals)
                for ind in individuals:
                    shared_fitness = ind['fitness'] / niche_size
                    pop_with_shared_fitness.append({'seq': ind['seq'], 'shared_fitness': shared_fitness})
            
            pop_with_shared_fitness.sort(key=lambda x: x['shared_fitness'], reverse=True)
            
            best_fitness_in_gen = pop_with_shared_fitness[0]['shared_fitness']
            t.set_postfix_str(f"Best Shared Fitness: {best_fitness_in_gen:.2f}, Niches: {len(species)}")

            # 4. Selezione basata sul Fitness Condiviso
            elites_count = int(self.population_size * 0.1)
            next_generation = [d['seq'] for d in pop_with_shared_fitness[:elites_count]]
            
            sequences_for_selection = [d['seq'] for d in pop_with_shared_fitness]
            selection_weights = [d['shared_fitness'] for d in pop_with_shared_fitness]
            min_fitness = min(selection_weights)
            selection_weights = [(s - min_fitness) + 1e-6 for s in selection_weights]

            # 5. Crossover e Mutazione (come prima)
            while len(next_generation) < self.population_size:
                parents = random.choices(sequences_for_selection, weights=selection_weights, k=2) if sum(selection_weights) > 0 else random.sample(sequences_for_selection, k=2)
                point = random.randint(1, self.solution_length - 1)
                child = parents[0][:point] + parents[1][point:]
                for i in range(len(child)):
                    if random.random() < self.mutation_rate: child[i] = random.choice(self.possible_actions)
                child = self._local_search_enhanced(initial_state, child)
                next_generation.append(child)
                
            population = next_generation

        t.close()
        print("No solution found, returning best sequence from the final population.")
        # Restituisce la migliore sequenza basata sul fitness condiviso
        return pop_with_shared_fitness[0]['seq']