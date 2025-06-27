import random
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Any, Dict
from tqdm import trange
import copy

#
# VERSIONE 11.0 - EVOLUZIONE DIRETTA (STATE-OF-THE-ART DA COMPETIZIONE)
#
# Questo agente implementa la strategia descritta nel paper come quella del vincitore
# della Keke AI Competition. È un algoritmo evolutivo che opera direttamente sulle
# sequenze di mosse, potenziato da due tecniche chiave:
# 1. PATH CACHING: Per velocizzare drasticamente la valutazione del fitness.
# 2. LOCAL SEARCH: Per rifinire e migliorare rapidamente le soluzioni promettenti.
#
class EVOLUTIONARYAgent(BaseAgent):
    """
    Agente a Evoluzione Diretta che ottimizza sequenze di mosse.
    """
    def __init__(self, population_size=40, generations=50, mutation_rate=0.15, solution_length=60, local_search_steps=3):
        """
        PARAMETRI OTTIMIZZATI PER LA VELOCITÀ:
        - population_size: Ridotta a 40 per diminuire il carico di lavoro per generazione.
        - generations: Ridotte a 50. Il totale è 40*50 = 2000 valutazioni, 10 volte meno di prima.
        - mutation_rate e local_search_steps: Leggermente aggiustati per il nuovo bilanciamento.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.solution_length = solution_length
        self.local_search_steps = local_search_steps
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        self.cache: Dict[Tuple[Direction, ...], GameState] = {}

    def _calculate_fitness(self, initial_state: GameState, sequence: List[Direction]) -> Tuple[float, int]:
        """Valuta una sequenza, usando la cache per la velocità."""
        best_intermediate_fitness = -1.0
        win_step = -1

        current_state = initial_state
        seq_tuple = tuple(sequence)

        # Controlla se l'intera sequenza è in cache
        if seq_tuple in self.cache:
            current_state = self.cache[seq_tuple]
        else:
            # Trova il prefisso più lungo già presente in cache
            for i in range(len(sequence), 0, -1):
                prefix = tuple(sequence[:i])
                if prefix in self.cache:
                    current_state = self.cache[prefix]
                    break
            else:
                i = 0

            # Simula solo la parte non in cache della sequenza
            for j in range(i, len(sequence)):
                current_state = advance_game_state(sequence[j], current_state.copy())
                # Salva in cache lo stato intermedio
                self.cache[tuple(sequence[:j+1])] = current_state

        # Ora calcola il fitness basandosi sullo stato finale della simulazione
        if check_win(current_state):
            # Per trovare il passo esatto della vittoria, dobbiamo ri-simulare (rapidamente)
            temp_state = initial_state
            for k, action in enumerate(sequence):
                temp_state = advance_game_state(action, temp_state.copy())
                if check_win(temp_state):
                    win_step = k + 1
                    break
            return 10000.0 - (win_step * 10), win_step

        if not current_state.players:
            return -10000.0, -1

        dist_to_goal = 100.0
        player_pos = [(p.x, p.y) for p in current_state.players]
        goal_pos = [(g.x, g.y) for g in current_state.winnables]
        if player_pos and goal_pos:
            dist_to_goal = min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)
        
        return 100.0 / (dist_to_goal + 1), -1

    def _local_search(self, initial_state: GameState, individual: List[Direction]) -> List[Direction]:
        """Tenta di migliorare un individuo con piccole modifiche locali."""
        current_best_seq = list(individual)
        current_best_fitness, _ = self._calculate_fitness(initial_state, current_best_seq)

        for _ in range(self.local_search_steps):
            # Prova a modificare una mossa casuale
            temp_seq = list(current_best_seq)
            idx_to_change = random.randrange(len(temp_seq))
            original_action = temp_seq[idx_to_change]
            temp_seq[idx_to_change] = random.choice(self.possible_actions)
            
            new_fitness, _ = self._calculate_fitness(initial_state, temp_seq)

            if new_fitness > current_best_fitness:
                current_best_fitness = new_fitness
                current_best_seq = temp_seq
            else:
                # Annulla la modifica se non ha migliorato
                temp_seq[idx_to_change] = original_action
        
        return current_best_seq

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        self.cache.clear() # Pulisci la cache per ogni nuovo livello
        population = [[random.choice(self.possible_actions) for _ in range(self.solution_length)] for _ in range(self.population_size)]

        for _ in trange(self.generations, desc="Direct Evolution (v11)"):
            pop_with_fitness = [(self._calculate_fitness(initial_state, seq), seq) for seq in population]

            # Controlla se una soluzione è stata trovata
            for (fitness, win_step), seq in pop_with_fitness:
                if win_step != -1:
                    return seq[:win_step]
            
            pop_with_fitness.sort(key=lambda x: x[0][0], reverse=True)
            
            # Selezione, Crossover e Mutazione...
            elites_count = int(self.population_size * 0.1)
            next_generation = [seq for (fitness, win_step), seq in pop_with_fitness[:elites_count]]
            
            fitness_scores = [f for (f, w), s in pop_with_fitness]
            min_fitness = min(fitness_scores)
            selection_weights = [(score - min_fitness) + 0.01 for score in fitness_scores]
            
            sequences_for_selection = [s for (f, w), s in pop_with_fitness]

            while len(next_generation) < self.population_size:
                if sum(selection_weights) > 0:
                    parents = random.choices(sequences_for_selection, weights=selection_weights, k=2)
                else: # Fallback
                    parents = random.sample(sequences_for_selection, k=2)
                
                # Operatore di Crossover/Mutazione ispirato dal paper 
                # Semplificato come uno scambio di segmenti
                point = random.randint(1, self.solution_length - 1)
                child = parents[0][:point] + parents[1][point:]
                
                # Mutazione standard
                for i in range(len(child)):
                    if random.random() < self.mutation_rate:
                        child[i] = random.choice(self.possible_actions)

                # Applica la ricerca locale per rifinire il nuovo individuo
                child = self._local_search(initial_state, child)
                next_generation.append(child)
            
            population = next_generation

        return []