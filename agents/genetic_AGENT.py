import random
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple
from tqdm import trange

#
# VERSIONE 8.0 - FINALE (STATO DELL'ARTE)
#
# Corretto il malinteso sui limiti: non ci sono limiti di iterazioni, ma la soluzione
# non può superare le 100 mosse.
# Questo agente implementa una strategia GA avanzata con:
# 1. LUNGHEZZA DELLA SOLUZIONE ADATTIVA: Inizia cercando soluzioni brevi e aumenta
#    la lunghezza solo se necessario. Massimizza l'efficienza.
# 2. FITNESS INTELLIGENTE: Valuta il progresso intermedio, non solo lo stato finale.
# 3. PARAMETRI ROBUSTI: Popolazione e generazioni adeguate per una vera evoluzione.
#
class GENETICAgent(BaseAgent):
    """
    Algoritmo Genetico con lunghezza della soluzione adattiva e fitness migliorata.
    Questa è una versione robusta e potente che rappresenta lo stato dell'arte.
    """

    def __init__(self, population_size=150, mutation_rate=0.1, elitism_pct=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_elites = int(population_size * elitism_pct)
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]

    def _calculate_fitness(self, initial_state: GameState, sequence: List[Direction]) -> float:
        current_state = initial_state.copy()
        best_intermediate_fitness = -1.0
        
        # Simula la sequenza di azioni
        for i, action in enumerate(sequence):
            current_state = advance_game_state(action, current_state.copy())

            # Se si vince, il fitness è altissimo e premia la brevità
            if check_win(current_state):
                return 10000.0 - (i * 5)

            # Se si perde, la penalità è massima
            if not current_state.players:
                return -10000.0

            # Calcola il fitness intermedio dello stato attuale
            current_fitness = 0.0
            player_pos = [(p.x, p.y) for p in current_state.players]
            if not player_pos: continue

            goal_pos = [(g.x, g.y) for g in current_state.winnables]
            if goal_pos:
                dist = min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)
                current_fitness += 10.0 / (dist + 1) # Ricompensa per la vicinanza all'obiettivo

            word_pos = [(w.x, w.y) for w in (current_state.words + current_state.is_connectors)]
            if word_pos:
                dist = min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in word_pos)
                current_fitness += 2.0 / (dist + 1) # Ricompensa minore per la vicinanza alle parole

            # Aggiorna il miglior fitness intermedio trovato finora
            if current_fitness > best_intermediate_fitness:
                best_intermediate_fitness = current_fitness
                
        return best_intermediate_fitness

    def _crossover(self, p1: List[Direction], p2: List[Direction]) -> Tuple[List[Direction], List[Direction]]:
        # Crossover a due punti per una maggiore variabilità
        if len(p1) <= 2: return p1, p2
        pt1, pt2 = sorted(random.sample(range(1, len(p1)), 2))
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        return c1, c2

    def _mutate(self, individual: List[Direction]) -> List[Direction]:
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.possible_actions)
        return individual

    def search(self, initial_state: GameState, iterations: int = 500) -> List[Direction]:
        max_solution_length = 100
        initial_length = 20
        length_increase = 20
        generations_per_stage = 150 # Numero di generazioni prima di aumentare la lunghezza

        current_length = initial_length
        population = [[random.choice(self.possible_actions) for _ in range(current_length)] for _ in range(self.population_size)]

        # Il ciclo esterno gestisce l'aumento della lunghezza della soluzione
        while current_length <= max_solution_length:
            stage_desc = f"Evolving (len={current_length})"
            
            # Il ciclo interno esegue le generazioni per la lunghezza corrente
            for _ in trange(generations_per_stage, desc=stage_desc):
                pop_with_fitness = [(self._calculate_fitness(initial_state, seq), seq) for seq in population]
                pop_with_fitness.sort(key=lambda x: x[0], reverse=True)
                
                best_fitness, best_sequence = pop_with_fitness[0]
                if best_fitness > 5000:
                    final_state = initial_state.copy()
                    for j, action in enumerate(best_sequence):
                        final_state = advance_game_state(action, final_state.copy())
                        if check_win(final_state):
                            return best_sequence[:j+1]

                next_generation = []
                elites = [seq for fitness, seq in pop_with_fitness[:self.num_elites]]
                next_generation.extend(elites)
                
                raw_fitness_scores = [f for f, seq in pop_with_fitness]
                min_fitness = min(raw_fitness_scores)
                selection_weights = [(score - min_fitness) + 0.01 for score in raw_fitness_scores]
                
                population_for_selection = [seq for fitness, seq in pop_with_fitness]
                
                while len(next_generation) < self.population_size:
                    if sum(selection_weights) > 0:
                        parents = random.choices(population_for_selection, weights=selection_weights, k=2)
                    else:
                        parents = random.sample(population_for_selection, k=2)

                    child1, child2 = self._crossover(parents[0], parents[1])
                    next_generation.append(self._mutate(child1))
                    if len(next_generation) < self.population_size:
                        next_generation.append(self._mutate(child2))
                
                population = next_generation

            # Se non è stata trovata una soluzione, aumenta la lunghezza e continua
            current_length += length_increase
            # Estendi gli individui esistenti con nuove mosse casuali
            for i in range(len(population)):
                population[i].extend([random.choice(self.possible_actions) for _ in range(length_increase)])

        return []