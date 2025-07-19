# hybrid_AGENT.py

import time
import math
import random
from typing import List, Optional, Tuple, Dict, Any

from base_agent import BaseAgent
from baba import (GameState, Direction, advance_game_state, check_win, GameObj, is_word)

# --- Funzioni Ausiliarie ---
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# --- Strutture Dati per MCTS ---
class MCTSNode:
    # ... (Classe MCTSNode invariata) ...
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]

    def ucb1_score(self, exploration_constant=1.41) -> float:
        if self.visits == 0: return float('inf')
        return (self.wins / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_best_child(self):
        return max(self.children, key=lambda c: c.ucb1_score())

    def expand(self):
        action = self.untried_actions.pop()
        next_state = advance_game_state(action, self.state.copy())
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

# --- L'Agente Ibrido ---
class MCTSAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.time_limit_per_subgoal = 20  # 20 secondi per ogni sotto-obiettivo
        self.simulation_depth = 30
        self.mcts_iterations = 50000 # Un numero alto, il tempo lo fermerà

    def search(self, initial_state: GameState, iterations: int = 50000) -> Optional[List[Direction]]:
        """
        Componente principale che orchestra il pianificatore e il risolutore.
        """
        self.mcts_iterations = iterations
        
        # 1. Il Pianificatore Strategico crea il piano
        plan = self._create_plan(initial_state)
        if not plan:
            print("❌ Pianificatore: Impossibile creare un piano.")
            return None
        
        print(f"✅ Piano Strategico Creato: {[subgoal['type'] for subgoal in plan]}")

        full_solution_path = []
        current_state = initial_state.copy()

        # 2. Esegui ogni passo del piano
        for i, subgoal in enumerate(plan):
            print(f"\n--- Inizio Subgoal {i+1}/{len(plan)}: {subgoal['type']} ---")
            
            # 3. Il Risolutore Tattico (MCTS) trova il percorso per il subgoal
            path_to_subgoal = self._solve_subgoal_with_mcts(current_state, subgoal)
            
            if path_to_subgoal is None:
                print(f"❌ MCTS Fallito sul subgoal {subgoal['type']}. Piano interrotto.")
                return None

            # Applica le mosse trovate e aggiorna lo stato
            for move in path_to_subgoal:
                current_state = advance_game_state(move, current_state.copy())
            
            full_solution_path.extend(path_to_subgoal)
            print(f"✅ Subgoal completato in {len(path_to_subgoal)} mosse.")

        return full_solution_path

    def _create_plan(self, state: GameState) -> List[Dict[str, Any]]:
        """
        Il Pianificatore Strategico. Analizza lo stato e crea una lista di sotto-obiettivi.
        Questa è una versione semplificata ma più intelligente.
        """
        plan = []
        
        # Logica di base: Assicurati YOU, poi PUSH necessari, poi WIN, poi vinci.
        if not any('-is-you' in r for r in state.rules):
            plan.append({'type': 'form_rule', 'target': 'baba-is-you'}) # Target semplificato
        
        # Esempio di rilevamento ostacoli: se ci sono rocce, assumi che serva PUSH
        if 'rock' in state.sort_phys and not any('rock-is-push' in r for r in state.rules):
             plan.append({'type': 'form_rule', 'target': 'rock-is-push'})

        if not any('-is-win' in r for r in state.rules):
            plan.append({'type': 'form_rule', 'target': 'flag-is-win'})
        
        plan.append({'type': 'reach_win_condition'})
        return plan

    def _solve_subgoal_with_mcts(self, start_state: GameState, subgoal: Dict) -> Optional[List[Direction]]:
        """
        Il Risolutore Tattico. Usa MCTS per raggiungere un singolo subgoal.
        """
        # Trova un percorso di mosse per raggiungere uno stato che soddisfi il subgoal
        # Questo è un problema di ricerca, che MCTS può risolvere.
        
        # Per semplicità, qui simuliamo un MCTS che ritorna un percorso
        # L'implementazione reale richiederebbe di eseguire MCTS mossa per mossa
        # fino a quando _is_subgoal_achieved(state, subgoal) è True.
        
        # Questa è la parte più complessa. L'MCTS come scritto prima va adattato
        # per fermarsi quando il subgoal è raggiunto, non solo alla vittoria finale.
        # Per ora, restituiamo un percorso fittizio per mostrare la logica.
        
        # --- Inizio Logica MCTS Adattata ---
        root = MCTSNode(state=start_state)
        start_time = time.time()
        
        for i in range(self.mcts_iterations):
            if time.time() - start_time > self.time_limit_per_subgoal:
                print(f"Timeout per il subgoal! Eseguite {i} iterazioni.")
                break

            node = root
            # 1. Selection
            while not node.untried_actions and node.children:
                node = node.select_best_child()
            # 2. Expansion
            if node.untried_actions:
                node = node.expand()
            # 3. Simulation
            # L'outcome ora è 1 se la simulazione RAGGIUNGE il subgoal
            outcome = self._simulate_playout_for_subgoal(node.state, subgoal)
            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += outcome
                node = node.parent
        
        # Ricostruisci il percorso migliore trovato finora
        # Cerca nel'albero il nodo con più vittorie che soddisfa il subgoal
        # Questa parte è complessa, per ora la semplifichiamo
        best_node = self.find_best_subgoal_node(root, subgoal)
        if best_node:
            path = []
            curr = best_node
            while curr.parent:
                path.append(curr.action)
                curr = curr.parent
            return path[::-1] # Inverti per avere il percorso corretto
            
        return None # Non è stato trovato un percorso

    def _simulate_playout_for_subgoal(self, state: GameState, subgoal: Dict) -> int:
        # Questa simulazione deve essere VELOCE. Usa una rappresentazione leggera dello stato.
        temp_state = state.copy()
        for _ in range(self.simulation_depth):
            if self._is_subgoal_achieved(temp_state, subgoal):
                return 1 # Successo!
            
            # Usa una mossa greedy basata su un'euristica per il subgoal
            # ... (logica di movimento greedy come nella versione precedente) ...
            
            # Scegli una mossa casuale per semplicità qui
            move = random.choice([Direction.Up, Direction.Down, Direction.Left, Direction.Right])
            temp_state = advance_game_state(move, temp_state)

        return 0 # Fallimento

    def _is_subgoal_achieved(self, state: GameState, subgoal: Dict) -> bool:
        if subgoal['type'] == 'form_rule':
            return subgoal['target'] in state.rules
        elif subgoal['type'] == 'reach_win_condition':
            return check_win(state)
        return False

    def find_best_subgoal_node(self, root: MCTSNode, subgoal: Dict) -> Optional[MCTSNode]:
        # Funzione per cercare nell'albero il miglior nodo che soddisfa il subgoal
        best_node = None
        best_score = -1
        
        nodes_to_visit = [root]
        visited_hashes = {str(root.state)}
        
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            
            if self._is_subgoal_achieved(current_node.state, subgoal):
                score = current_node.wins / current_node.visits if current_node.visits > 0 else 0
                if score > best_score:
                    best_score = score
                    best_node = current_node
            
            for child in current_node.children:
                h = str(child.state)
                if h not in visited_hashes:
                    nodes_to_visit.append(child)
                    visited_hashes.add(h)
                    
        return best_node