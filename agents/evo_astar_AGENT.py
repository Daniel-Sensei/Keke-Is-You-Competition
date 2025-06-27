import random
from collections import deque
from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Any, Dict, Set
from tqdm import trange
import heapq

class EVO_ASTARAgent(BaseAgent):
    """
    Agente evolutivo ottimizzato per velocità mantenendo l'efficacia.
    
    Ottimizzazioni principali:
    1. Cache intelligente per stati e percorsi
    2. Pathfinding A* con euristica migliorata
    3. Valutazione lazy della fitness
    4. Popolazione adattiva
    5. Terminazione anticipata intelligente
    """

    def __init__(self, population_size=50, generations=100, mutation_rate=0.3, 
                 max_plan_length=6, elitism_pct=0.15, max_search_nodes=200):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_plan_length = max_plan_length
        self.elitism_pct = elitism_pct
        self.num_elites = int(self.population_size * self.elitism_pct)
        self.max_search_nodes = max_search_nodes
        
        # Cache per ottimizzare le ricerche ripetute
        self.path_cache: Dict[str, List[Direction]] = {}
        self.state_cache: Dict[str, GameState] = {}
        self.fitness_cache: Dict[str, Tuple[float, List[Direction]]] = {}
        
        # Statistiche per terminazione anticipata
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def _state_to_key(self, state: GameState) -> str:
        """Genera una chiave univoca per lo stato del gioco."""
        return str(state)

    def _plan_to_key(self, plan: List[Any]) -> str:
        """Genera una chiave univoca per un piano."""
        return str([(obj.x, obj.y, type(obj).__name__) for obj in plan])

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcola la distanza di Manhattan tra due posizioni."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_path_astar_optimized(self, start_state: GameState, target_pos: Tuple[int, int]) -> List[Direction]:
        """
        Pathfinder A* ottimizzato con cache e euristica migliorata.
        """
        if not start_state.players:
            return None

        # Controlla la cache
        cache_key = f"{self._state_to_key(start_state)}_{target_pos}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        start_pos = (start_state.players[0].x, start_state.players[0].y)
        
        # Euristica: se il target è molto lontano, probabilmente non è raggiungibile rapidamente
        if self._manhattan_distance(start_pos, target_pos) > 30:
            self.path_cache[cache_key] = None
            return None

        # A* con priorità basata su costo + euristica
        # Usiamo un counter per evitare conflitti di confronto tra GameState
        counter = 0
        heap = [(0, 0, counter, start_state.copy(), [])]  # (f_score, g_score, tie_breaker, state, path)
        visited: Set[str] = set()
        nodes_explored = 0

        while heap and nodes_explored < self.max_search_nodes:
            f_score, g_score, _, current_state, path = heapq.heappop(heap)
            nodes_explored += 1

            state_key = self._state_to_key(current_state)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Controlla se abbiamo raggiunto l'obiettivo
            for player in current_state.players:
                if (player.x, player.y) == target_pos:
                    self.path_cache[cache_key] = path
                    return path

            # Esplora le azioni possibili
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(action, current_state.copy())
                if not next_state.players:
                    continue
                
                next_state_key = self._state_to_key(next_state)
                if next_state_key in visited:
                    continue

                new_g_score = g_score + 1
                player_pos = (next_state.players[0].x, next_state.players[0].y)
                h_score = self._manhattan_distance(player_pos, target_pos)
                f_score = new_g_score + h_score

                counter += 1
                heapq.heappush(heap, (f_score, new_g_score, counter, next_state, path + [action]))

        # Nessun percorso trovato
        self.path_cache[cache_key] = None
        return None

    def _get_potential_goals_smart(self, state: GameState) -> List[Any]:
        """
        Seleziona obiettivi potenziali in modo intelligente.
        Prioritizza oggetti vicini e probabilmente importanti.
        """
        all_objects = state.phys + state.words + state.is_connectors
        
        if not state.players or not all_objects:
            return all_objects
        
        player_pos = (state.players[0].x, state.players[0].y)
        
        # Ordina per distanza di Manhattan
        objects_with_distance = [
            (obj, self._manhattan_distance(player_pos, (obj.x, obj.y)))
            for obj in all_objects
        ]
        objects_with_distance.sort(key=lambda x: x[1])
        
        # Prende i primi N oggetti più vicini (aumentato)
        max_candidates = min(20, len(objects_with_distance))
        return [obj for obj, _ in objects_with_distance[:max_candidates]]

    def _calculate_fitness_lazy(self, initial_state: GameState, plan: List[Any]) -> Tuple[float, List[Direction]]:
        """
        Calcola la fitness con valutazione lazy e cache.
        """
        # Controlla la cache
        plan_key = self._plan_to_key(plan)
        state_key = self._state_to_key(initial_state)
        cache_key = f"{state_key}_{plan_key}"
        
        if cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]

        current_state = initial_state.copy()
        full_path = []
        goals_achieved = 0
        total_distance = 0

        for i, goal_obj in enumerate(plan):
            target_pos = (goal_obj.x, goal_obj.y)
            
            # Calcola distanza prima di cercare il percorso
            if current_state.players:
                player_pos = (current_state.players[0].x, current_state.players[0].y)
                distance = self._manhattan_distance(player_pos, target_pos)
                total_distance += distance
                
                # Se la distanza è troppo grande, interrompe la valutazione
                if distance > 35:
                    break
            
            path_to_goal = self._find_path_astar_optimized(current_state, target_pos)

            if path_to_goal:
                goals_achieved += 1
                full_path.extend(path_to_goal)
                
                # Simula solo se necessario
                for action in path_to_goal:
                    current_state = advance_game_state(action, current_state)
                    if not current_state.players:
                        result = (-1000.0, [])
                        self.fitness_cache[cache_key] = result
                        return result
            else:
                # Penalizza piani non eseguibili
                break

        # Calcola fitness con più fattori
        fitness = (goals_achieved * 200 - 
                  len(full_path) * 2 - 
                  total_distance * 0.5 -
                  (len(plan) - goals_achieved) * 50)  # Penalizza obiettivi non raggiunti

        # Bonus per vittoria
        if check_win(current_state):
            fitness += 15000.0

        result = (fitness, full_path)
        self.fitness_cache[cache_key] = result
        return result

    def _should_terminate_early(self, current_best_fitness: float) -> bool:
        """
        Determina se terminare l'evoluzione anticipatamente.
        Resa meno aggressiva per permettere più esplorazione.
        """
        self.best_fitness_history.append(current_best_fitness)
        
        # Mantiene solo le ultime 15 generazioni (aumentato)
        if len(self.best_fitness_history) > 15:
            self.best_fitness_history.pop(0)
        
        # Se abbiamo trovato una soluzione vincente
        if current_best_fitness > 12000:
            return True
        
        # Terminazione per stagnazione solo dopo più generazioni
        if len(self.best_fitness_history) >= 12:
            recent_improvements = []
            for i in range(len(self.best_fitness_history) - 1):
                if self.best_fitness_history[i+1] > self.best_fitness_history[i]:
                    recent_improvements.append(i)
            
            # Se non ci sono stati miglioramenti nelle ultime 12 generazioni
            if not recent_improvements:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
            # Termina solo dopo 8 generazioni di vera stagnazione
            if self.stagnation_counter >= 8:
                return True
        
        return False

    def _create_initial_population_smart(self, potential_goals: List[Any]) -> List[List[Any]]:
        """
        Crea una popolazione iniziale più intelligente con piani più lunghi.
        """
        population = []
        
        # Una parte della popolazione ha piani corti (successo rapido)
        for _ in range(self.population_size // 4):
            length = random.randint(2, max(2, self.max_plan_length // 2))
            population.append(random.choices(potential_goals, k=length))
        
        # Una parte ha piani di lunghezza media
        for _ in range(self.population_size // 4):
            length = random.randint(self.max_plan_length // 2, self.max_plan_length)
            population.append(random.choices(potential_goals, k=length))
        
        # Una parte ha piani lunghi (per livelli complessi)
        for _ in range(self.population_size // 4):
            length = self.max_plan_length
            population.append(random.choices(potential_goals, k=length))
        
        # Resto con piani che favoriscono oggetti vicini
        for _ in range(self.population_size - len(population)):
            length = random.randint(3, self.max_plan_length)
            # Maggiore probabilità per i primi oggetti (più vicini)
            weights = [max(1, len(potential_goals) - i) for i in range(len(potential_goals))]
            population.append(random.choices(potential_goals, weights=weights, k=length))
        
        return population

    def search(self, initial_state: GameState, iterations: int = 0) -> List[Direction]:
        # Reset cache e statistiche
        self.path_cache.clear()
        self.fitness_cache.clear()
        self.stagnation_counter = 0
        self.best_fitness_history.clear()
        
        generations = self.generations if iterations <= 0 else iterations
        
        potential_goals = self._get_potential_goals_smart(initial_state)
        if not potential_goals:
            return []

        population = self._create_initial_population_smart(potential_goals)
        
        best_solution_so_far = []
        highest_fitness = -float('inf')

        for generation in trange(generations, desc="Optimized Evolution"):
            # Valuta la popolazione
            pop_with_results = []
            for plan in population:
                fitness, path = self._calculate_fitness_lazy(initial_state, plan)
                pop_with_results.append(((fitness, path), plan))
            
            # Ordina per fitness
            pop_with_results.sort(key=lambda x: x[0][0], reverse=True)

            # Aggiorna la migliore soluzione
            current_best_fitness, current_best_path = pop_with_results[0][0]
            if current_best_fitness > highest_fitness:
                highest_fitness = current_best_fitness
                best_solution_so_far = current_best_path

            # Terminazione anticipata
            if self._should_terminate_early(current_best_fitness):
                break

            # Evoluzione della popolazione
            elites = [plan for (_, _), plan in pop_with_results[:self.num_elites]]
            
            # Selezione con tournament selection (più veloce)
            next_generation = elites[:]
            
            while len(next_generation) < self.population_size:
                # Tournament selection
                tournament_size = 5
                tournament = random.sample(pop_with_results, min(tournament_size, len(pop_with_results)))
                parent1 = max(tournament, key=lambda x: x[0][0])[1]
                
                tournament = random.sample(pop_with_results, min(tournament_size, len(pop_with_results)))
                parent2 = max(tournament, key=lambda x: x[0][0])[1]
                
                # Crossover
                if len(parent1) > 1 and len(parent2) > 1:
                    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                    child = parent1[:point] + parent2[point:]
                else:
                    child = parent1[:]
                
                # Mutazione
                for i in range(len(child)):
                    if random.random() < self.mutation_rate:
                        child[i] = random.choice(potential_goals)
                
                # Mutazione strutturale (aggiunge/rimuove elementi)
                if random.random() < 0.1:
                    if len(child) > 1 and random.random() < 0.5:
                        child.pop(random.randint(0, len(child) - 1))
                    elif len(child) < self.max_plan_length:
                        child.append(random.choice(potential_goals))
                
                next_generation.append(child)

            population = next_generation[:self.population_size]
            
            # Adatta parametri dinamicamente (meno frequente)
            if generation > 0 and generation % 20 == 0:
                # Aumenta mutation rate se c'è stagnazione
                if self.stagnation_counter > 3:
                    self.mutation_rate = min(0.6, self.mutation_rate * 1.2)
                else:
                    self.mutation_rate = max(0.15, self.mutation_rate * 0.9)

        return best_solution_so_far