import heapq
import itertools
import time
from typing import List, Dict, Tuple, Set, Optional

from base_agent import BaseAgent
from baba import (GameState, Direction, GameObj, GameObjectType, advance_game_state,
                  check_win, name_to_character)

# --- Funzioni Ausiliarie ---

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_word_objects(state: GameState, word_name: str) -> List[GameObj]:
    return [w for w in state.words if w.name == word_name]

def analyze_current_rules(state: GameState) -> Dict[str, List[str]]:
    rule_analysis = {'you_rules': [], 'win_rules': [], 'other_rules': []}
    for rule in state.rules:
        if '-is-you' in rule:
            rule_analysis['you_rules'].append(rule)
        elif '-is-win' in rule:
            rule_analysis['win_rules'].append(rule)
        else:
            rule_analysis['other_rules'].append(rule)
    return rule_analysis

# --- Classe per HeapQ Entry ---
class HeapQEntry:
    def __init__(self, priority: float, g_score: int, tie_breaker: int, actions: List[Direction], state: GameState):
        self.priority = priority
        self.g_score = g_score
        self.tie_breaker = tie_breaker
        self.actions = actions
        self.state = state

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.g_score != other.g_score:
            return self.g_score > other.g_score
        return self.tie_breaker < other.tie_breaker

# --- Agente A* con Risoluzione di Sotto-Problemi ---
class A_STAR_SUB_ADATTIVOAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 200000
        self.counter = itertools.count()
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.0
        self.P_NO_WIN_RULE = 100
        self.P_NO_YOU_RULE = 500
        self.max_path_length = 60
        self.subgoal_enabled = True  # Flag per abilitare/disabilitare i sotto-obiettivi
        self.subgoal_max_moves = 30  # Limite mosse per sotto-obiettivo

    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        start_time = time.time()
        
        # Prova prima con l'approccio dei sotto-obiettivi
        if self.subgoal_enabled:
            print("🎯 Tentativo con risoluzione di sotto-obiettivi...")
            result = self._search_with_subgoals(initial_state, start_time)
            if result:
                return result
            print("⚠️ Fallimento con sotto-obiettivi, fallback a A* standard...")
        
        # Fallback a A* standard
        return self._search_standard(initial_state, start_time)

    def _search_with_subgoals(self, initial_state: GameState, start_time: float) -> Optional[List[Direction]]:
        """Ricerca con decomposizione in sotto-obiettivi"""
        goal_stack = self._identify_goal_stack(initial_state)
        total_plan: List[Direction] = []
        current_state = initial_state.copy()
        
        for subgoal in goal_stack:
            print(f"🎯 Risolvendo sotto-obiettivo: {subgoal['name']}")
            
            # Controlla se il sotto-obiettivo è già soddisfatto
            if self._goal_reached(current_state, subgoal):
                print(f"✅ Sotto-obiettivo '{subgoal['name']}' già soddisfatto")
                continue
            
            plan = self._search_for_subgoal(current_state, subgoal, start_time)
            if not plan:
                print(f"❌ Fallito su obiettivo: {subgoal['name']}")
                return None
            
            # Applica le mosse del piano
            for move in plan:
                current_state = advance_game_state(move, current_state.copy())
            total_plan.extend(plan)
            
            print(f"✅ Completato sotto-obiettivo: {subgoal['name']} con {len(plan)} mosse")
            
            # Controlla se abbiamo vinto
            if check_win(current_state):
                print("🏆 Vittoria raggiunta!")
                return total_plan
        
        # Verifica finale se abbiamo vinto
        return total_plan if check_win(current_state) else None

    def _identify_goal_stack(self, state: GameState) -> List[Dict]:
        """Identifica la sequenza di sotto-obiettivi da risolvere"""
        rule_analysis = analyze_current_rules(state)
        goals = []
        
        # 1. Prima assicurarsi di avere una regola YOU
        if not rule_analysis['you_rules']:
            goals.append({
                'name': 'CREA_REGOLA_YOU',
                'type': 'rule_formation',
                'target_rule': 'you'
            })
        
        # 2. Poi assicurarsi di avere una regola WIN
        if not rule_analysis['win_rules']:
            goals.append({
                'name': 'CREA_REGOLA_WIN',
                'type': 'rule_formation',
                'target_rule': 'win'
            })
        
        # 3. Infine raggiungere l'oggetto vincente
        if rule_analysis['you_rules'] and rule_analysis['win_rules']:
            goals.append({
                'name': 'RAGGIUNGI_OGGETTO_VINCENTE',
                'type': 'reach_win',
                'target_rule': None
            })
        
        return goals

    def _search_for_subgoal(self, start_state: GameState, subgoal: Dict, start_time: float) -> Optional[List[Direction]]:
        """Ricerca A* focalizzata su un singolo sotto-obiettivo"""
        time_limit = 400  # secondi
        
        # Seleziona l'euristica appropriata per il sotto-obiettivo
        if subgoal['type'] == 'rule_formation':
            h_func = lambda s: self._heuristic_rule_formation(s, subgoal['target_rule'])
        elif subgoal['type'] == 'reach_win':
            h_func = lambda s: self._heuristic_reach_win(s)
        else:
            h_func = self._heuristic
        
        open_set = []
        closed_set = {}
        self.counter = itertools.count()
        
        start_h = h_func(start_state)
        if start_h == float('inf'):
            return None
        
        heapq.heappush(open_set, HeapQEntry(start_h, 0, next(self.counter), [], start_state))
        initial_hash = self._get_state_hash(start_state)
        if initial_hash:
            closed_set[initial_hash] = 0
        
        for _ in range(self.max_iterations):
            if time.time() - start_time > time_limit:
                print(f"⏰ Timeout per sotto-obiettivo: {subgoal['name']}")
                return None
            
            if not open_set:
                break
            
            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state
            
            # Controlla se il sotto-obiettivo è raggiunto
            if self._goal_reached(current_state, subgoal):
                return actions
            
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.subgoal_max_moves:
                    continue
                
                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)
                
                if state_hash and new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue
                
                h_score = h_func(next_state)
                if h_score == float('inf'):
                    continue
                
                if state_hash:
                    closed_set[state_hash] = new_g_score
                
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))
        
        return None

    def _goal_reached(self, state: GameState, subgoal: Dict) -> bool:
        """Controlla se un sotto-obiettivo è stato raggiunto"""
        if subgoal['type'] == 'rule_formation':
            rules = analyze_current_rules(state)
            if subgoal['target_rule'] == 'you':
                return bool(rules['you_rules'])
            elif subgoal['target_rule'] == 'win':
                return bool(rules['win_rules'])
        elif subgoal['type'] == 'reach_win':
            return check_win(state)
        
        return False

    def _heuristic_rule_formation(self, state: GameState, target_property: str) -> float:
        """Euristica specifica per la formazione di regole"""
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        property_words = find_word_objects(state, target_property)
        
        if not nouns or not is_words or not property_words:
            return float('inf')
        
        min_formation_dist = float('inf')
        
        for noun in nouns:
            if noun.name == target_property:
                continue
                
            # Distanza dal sostantivo alla parola "is"
            dist_to_is = min([manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words])
            
            # Distanza dal sostantivo alla proprietà target
            dist_to_prop = min([manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words])
            
            current_dist = dist_to_is + dist_to_prop
            min_formation_dist = min(min_formation_dist, current_dist)
        
        # Aggiungi la distanza del giocatore per spingere le parole
        if state.players and min_formation_dist != float('inf'):
            player_dist = min([manhattan_distance((p.x, p.y), (n.x, n.y)) for p in state.players for n in nouns])
            return (min_formation_dist + player_dist) * self.W_FORM_RULE
        
        return min_formation_dist * self.W_FORM_RULE

    def _heuristic_reach_win(self, state: GameState) -> float:
        """Euristica specifica per raggiungere oggetti vincenti"""
        if check_win(state):
            return 0
        
        if not state.players or not state.winnables:
            return float('inf')
        
        min_dist = float('inf')
        for player in state.players:
            for winnable in state.winnables:
                dist = manhattan_distance((player.x, player.y), (winnable.x, winnable.y))
                min_dist = min(min_dist, dist)
        
        return min_dist * self.W_REACH_WIN

    def _search_standard(self, initial_state: GameState, start_time: float) -> Optional[List[Direction]]:
        """Ricerca A* standard (originale)"""
        time_limit = 400  # secondi
        
        start_h = self._heuristic(initial_state)
        if start_h == float('inf'):
            return None
        
        open_set = [HeapQEntry(start_h, 0, next(self.counter), [], initial_state)]
        closed_set = {}
        
        initial_hash = self._get_state_hash(initial_state)
        if initial_hash:
            closed_set[initial_hash] = 0
        
        for _ in range(self.max_iterations):
            if time.time() - start_time > time_limit:
                print("⏰ Tempo massimo raggiunto per A* standard.")
                return None
            
            if not open_set:
                break
            
            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state
            
            if check_win(current_state):
                return actions
            
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.max_path_length:
                    continue
                
                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)
                
                if state_hash and new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue
                
                h_score = self._heuristic(next_state)
                if h_score == float('inf'):
                    continue
                
                if state_hash:
                    closed_set[state_hash] = new_g_score
                
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))
        
        print("⚠️ Nessuna soluzione trovata entro il limite di mosse.")
        return None

    def _heuristic(self, state: GameState) -> float:
        """Euristica originale completa"""
        if check_win(state):
            return 0
        if not state.players:
            return float('inf')
        
        rule_analysis = analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        has_win = bool(rule_analysis['win_rules'])
        
        if has_you and has_win:
            if state.winnables:
                min_dist = float('inf')
                for player in state.players:
                    for winnable in state.winnables:
                        dist = manhattan_distance((player.x, player.y), (winnable.x, winnable.y))
                        min_dist = min(min_dist, dist)
                return min_dist * self.W_REACH_WIN
            else:
                winnable_noun = rule_analysis['win_rules'][0].split('-is-')[0]
                cost = self._estimate_rule_formation_cost(state, winnable_noun)
                if cost == float('inf'):
                    return float('inf')
                return self.P_NO_WIN_RULE + cost
        
        if not has_you:
            cost = self._estimate_rule_formation_cost(state, 'you')
            return self.P_NO_YOU_RULE + cost
        
        if not has_win:
            cost = self._estimate_rule_formation_cost(state, 'win')
            return self.P_NO_WIN_RULE + cost
        
        return float('inf')

    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        """Stima del costo per formare una regola (metodo originale)"""
        nouns = [w for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
        is_words = find_word_objects(state, 'is')
        property_words = find_word_objects(state, target_property_name)
        
        if not nouns or not is_words or not property_words:
            return float('inf')
        
        min_formation_dist = float('inf')
        
        for noun in nouns:
            if noun.name == target_property_name:
                continue
            dist_to_is = min([manhattan_distance((noun.x, noun.y), (iw.x, iw.y)) for iw in is_words])
            dist_to_prop = min([manhattan_distance((noun.x, noun.y), (pw.x, pw.y)) for pw in property_words])
            current_dist = dist_to_is + dist_to_prop
            min_formation_dist = min(min_formation_dist, current_dist)
        
        if state.players:
            player_dist = min([manhattan_distance((p.x, p.y), (n.x, n.y)) for p in state.players for n in nouns])
            return (min_formation_dist + player_dist) * self.W_FORM_RULE
        
        return min_formation_dist * self.W_FORM_RULE

    def _get_state_hash(self, state: GameState) -> Optional[str]:
        """Genera hash dello stato per evitare cicli"""
        if not state.players:
            return "NO_PLAYERS_STATE"
        
        components = []
        
        player_pos = sorted([(p.x, p.y) for p in state.players])
        components.append(f"P:{','.join(map(str, player_pos))}")
        
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        components.append(f"W:{','.join(map(str, word_pos))}")
        
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        components.append(f"O:{','.join(map(str, phys_pos))}")
        
        return "|".join(components)

