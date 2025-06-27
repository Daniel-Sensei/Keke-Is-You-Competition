import random
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Any, Dict
from tqdm import trange
import copy

#
# VERSIONE 12.2 - INIZIALIZZAZIONE DIVERSIFICATA E MEMORIA GLOBALE
#
# Questa versione corregge il difetto critico della v12.1.
# 1. INIZIALIZZAZIONE DIVERSIFICATA: La popolazione di ogni ciclo non ha più una lunghezza
#    fissa, ma una lunghezza casuale variabile tra un minimo e il massimo del ciclo corrente.
#    Questo permette di trovare soluzioni brevi in modo esponenzialmente più efficace.
# 2. MEMORIA DELLA SOLUZIONE MIGLIORE: L'agente ora tiene traccia della soluzione
#    vincente più breve trovata *attraverso tutti i cicli*, evitando di fermarsi
#    con una soluzione più lunga trovata in un ciclo successivo.
#
class PROGRESSIVEAgent(BaseAgent):
    """
    Agente evolutivo con inizializzazione a lunghezza diversificata per una ricerca
    più efficiente delle soluzioni brevi.
    """
    def __init__(self,
                 initial_length: int = 8,
                 length_increment: int = 4,
                 max_length: int = 60,
                 generations_per_step: int = 40, # Aumentiamo un po' per dare tempo di convergere
                 population_size: int = 40,
                 mutation_rate: float = 0.25, # Leggero aumento per favorire l'esplorazione
                 local_search_steps: int = 3):
        
        self.initial_length = initial_length
        self.length_increment = length_increment
        self.max_length = max_length
        self.generations_per_step = generations_per_step
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.local_search_steps = local_search_steps
        self.possible_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        self.cache: Dict[Tuple[Direction, ...], GameState] = {}

    def _calculate_fitness(self, initial_state: GameState, sequence: List[Direction]) -> Tuple[float, int]:
        # ... (Nessuna modifica qui, ma la commenterò per chiarezza)
        if not sequence: return -10000.0, -1
        seq_tuple = tuple(sequence)
        if seq_tuple in self.cache:
            final_state = self.cache[seq_tuple]
        else:
            current_state = initial_state
            start_index = 0
            for i in range(len(sequence), 0, -1):
                prefix = tuple(sequence[:i])
                if prefix in self.cache:
                    current_state = self.cache[prefix]
                    start_index = i
                    break
            for j in range(start_index, len(sequence)):
                current_state = advance_game_state(sequence[j], current_state.copy())
                self.cache[tuple(sequence[:j+1])] = current_state
            final_state = current_state

        if check_win(final_state):
            return 10000.0 - (len(sequence) * 20), len(sequence)
        if not final_state.players:
            return -10000.0, -1

        dist_to_goal = 100.0
        player_pos = [(p.x, p.y) for p in final_state.players]
        goal_pos = [(g.x, g.y) for g in final_state.winnables]
        if player_pos and goal_pos:
            dist_to_goal = min(abs(p[0] - g[0]) + abs(p[1] - g[1]) for p in player_pos for g in goal_pos)
        
        base_fitness = 100.0 / (dist_to_goal + 1)
        # MODIFICA: Aumentata leggermente la penalità per la lunghezza per guidare meglio la ricerca
        length_penalty = len(sequence) * 0.2
        return base_fitness - length_penalty, -1

    def _mutate(self, sequence: List[Direction], max_len: int) -> List[Direction]:
        # ... (Questo metodo va già bene)
        if not sequence: return [random.choice(self.possible_actions)]
        mutation_type = random.random()
        new_sequence = list(sequence)
        if mutation_type < 0.6:
            idx = random.randrange(len(new_sequence))
            new_sequence[idx] = random.choice(self.possible_actions)
        elif mutation_type < 0.8:
            if len(new_sequence) < max_len:
                idx = random.randrange(len(new_sequence) + 1)
                new_sequence.insert(idx, random.choice(self.possible_actions))
        else:
            if len(new_sequence) > 1:
                idx = random.randrange(len(new_sequence))
                new_sequence.pop(idx)
        return new_sequence

    def _local_search(self, initial_state: GameState, individual: List[Direction]) -> List[Direction]:
        # ... (Questo metodo va già bene)
        current_best_seq = list(individual)
        current_best_fitness, _ = self._calculate_fitness(initial_state, current_best_seq)
        for _ in range(self.local_search_steps):
            temp_seq = list(current_best_seq)
            if not temp_seq: continue
            idx_to_change = random.randrange(len(temp_seq))
            original_action = temp_seq[idx_to_change]
            temp_seq[idx_to_change] = random.choice(self.possible_actions)
            new_fitness, _ = self._calculate_fitness(initial_state, temp_seq)
            if new_fitness > current_best_fitness:
                current_best_fitness = new_fitness
                current_best_seq = temp_seq
            else:
                temp_seq[idx_to_change] = original_action
        return current_best_seq

    def _run_evolution_step(self, initial_state: GameState, min_len: int, max_len: int) -> List[Direction]:
        """Esegue un ciclo di evoluzione con popolazione a lunghezza variabile."""
        
        # --- MODIFICA CHIAVE ---
        # La popolazione ora viene creata con lunghezze casuali nell'intervallo [min_len, max_len]
        population = []
        for _ in range(self.population_size):
            # Assicuriamo che min_len non sia più grande di max_len
            current_min = min(min_len, max_len)
            l = random.randint(current_min, max_len)
            population.append([random.choice(self.possible_actions) for _ in range(l)])

        best_solution_in_step = []

        for _ in trange(self.generations_per_step, desc=f"Evolving (len ∈ [{min_len}, {max_len}])"):
            pop_with_fitness = [(self._calculate_fitness(initial_state, seq), seq) for seq in population]

            for (fitness, win_step), seq in pop_with_fitness:
                if win_step != -1:
                    # Trovata una soluzione vincente. Controlliamo se è la migliore di questo step.
                    if not best_solution_in_step or len(seq) < len(best_solution_in_step):
                        best_solution_in_step = seq
            
            # Se abbiamo trovato una soluzione in questo step, continuiamo a evolvere
            # per vedere se ne troviamo una ancora più corta, ma non c'è bisogno di continuare all'infinito.
            # Per ora, restituiamo la migliore trovata alla fine del ciclo di generazioni.

            pop_with_fitness.sort(key=lambda x: x[0][0], reverse=True)
            
            elites_count = int(self.population_size * 0.1)
            next_generation = [seq for (_, _), seq in pop_with_fitness[:elites_count]]
            
            fitness_scores = [f for (f, w), s in pop_with_fitness]
            min_fitness = min(fitness_scores) if fitness_scores else 0
            selection_weights = [(score - min_fitness) + 0.01 for score in fitness_scores]
            
            sequences_for_selection = [s for (f, w), s in pop_with_fitness]

            while len(next_generation) < self.population_size:
                if sum(selection_weights) > 0:
                    parents = random.choices(sequences_for_selection, weights=selection_weights, k=2)
                else:
                    parents = random.sample(sequences_for_selection, k=2)
                
                point = random.randint(1, min(len(p) for p in parents) - 1) if all(len(p) > 1 for p in parents) else 0
                child = parents[0][:point] + parents[1][point:]
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, max_len + self.length_increment)
                
                if self.local_search_steps > 0:
                    child = self._local_search(initial_state, child)
                next_generation.append(child)
            
            population = next_generation

        return best_solution_in_step

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        self.cache.clear()
        current_length = self.initial_length
        
        # --- MODIFICA CHIAVE ---
        # Teniamo traccia della migliore soluzione trovata in assoluto
        best_solution_found = []
        
        while current_length <= self.max_length:
            print(f"\n[Progressive Search] Searching for solutions up to length {current_length}...")
            
            # Passiamo un range di lunghezze per l'inizializzazione
            min_len_for_step = 1 # Partiamo sempre da soluzioni molto brevi
            solution_in_step = self._run_evolution_step(initial_state, min_len_for_step, current_length)
            
            if solution_in_step:
                print(f"✨ Found a winning solution of length {len(solution_in_step)} in this step.")
                # Controlliamo se è la migliore trovata finora
                if not best_solution_found or len(solution_in_step) < len(best_solution_found):
                    best_solution_found = solution_in_step
                    print(f"🏆 New best global solution! Length: {len(best_solution_found)}")

            # Invece di fermarci, potremmo continuare per un altro ciclo per vedere se si trova di meglio.
            # Per ora, ci fermiamo appena troviamo una soluzione e abbiamo completato il ciclo.
            # Se vuoi essere più esaustivo, puoi rimuovere il break e lasciare che completi tutti i cicli.
            if best_solution_found:
                 # Potremmo decidere di fare un altro ciclo per sicurezza, o fermarci.
                 # Per velocità, fermarsi è meglio.
                 break
            
            current_length += self.length_increment
            
        if best_solution_found:
            print(f"\nReturning best solution found with length {len(best_solution_found)}.")
            return best_solution_found
        else:
            print("\nNo solution found within the maximum configured length.")
            return []