import heapq
import itertools
import time
from typing import List, Dict, Tuple, Set, Optional
from copy import deepcopy

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

def get_all_possible_rules(state: GameState) -> List[str]:
    """Restituisce tutte le regole che possono teoricamente essere formate"""
    nouns = [w.name for w in state.words if w.object_type == GameObjectType.Word and w.name != 'is']
    properties = ['you', 'win', 'push', 'stop', 'sink', 'hot', 'melt', 'open', 'shut', 'move']
    
    possible_rules = []
    available_props = [w.name for w in state.words if w.name in properties]
    
    for noun in set(nouns):
        for prop in available_props:
            if noun != prop:
                possible_rules.append(f"{noun}-is-{prop}")
    
    return possible_rules

def can_form_rule(state: GameState, rule: str) -> bool:
    """Verifica se una regola può essere formata con le parole disponibili"""
    parts = rule.split('-is-')
    if len(parts) != 2:
        return False
    
    noun_name, prop_name = parts[0], parts[1]
    
    # Controlla se esistono le parole necessarie
    noun_words = find_word_objects(state, noun_name)
    is_words = find_word_objects(state, 'is')
    prop_words = find_word_objects(state, prop_name)
    
    return len(noun_words) > 0 and len(is_words) > 0 and len(prop_words) > 0

def is_rule_critical(rule: str) -> bool:
    """Determina se una regola è critica (non dovrebbe essere distrutta)"""
    critical_patterns = ['-is-you', '-is-win', '-is-push']
    return any(pattern in rule for pattern in critical_patterns)

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

class HIERARCHICALAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.max_iterations = 50000
        self.counter = itertools.count()
        self.W_REACH_WIN = 1.0
        self.W_FORM_RULE = 1.0
        self.P_NO_WIN_RULE = 100
        self.P_NO_YOU_RULE = 500
        self.max_path_length = 25
        self.subgoal_max_path = 15  # Limite per sotto-obiettivi
        
    def search(self, initial_state: GameState, iterations: int = 10000) -> Optional[List[Direction]]:
        """
        Strategia gerarchica:
        1. Analizza se servono regole aggiuntive
        2. Forma le regole necessarie
        3. Risolve il problema principale
        """
        print("🔍 Avvio ricerca gerarchica...")
        
        # Analizza lo stato iniziale
        rule_analysis = analyze_current_rules(initial_state)
        missing_rules = self._identify_missing_rules(initial_state, rule_analysis)
        
        if missing_rules:
            print(f"📋 Regole mancanti identificate: {missing_rules}")
            return self._solve_with_subgoals(initial_state, missing_rules)
        else:
            print("✅ Tutte le regole necessarie sono presenti, avvio ricerca diretta")
            return self._basic_search(initial_state)
    
    def _identify_missing_rules(self, state: GameState, rule_analysis: Dict) -> List[str]:
        """Identifica le regole mancanti necessarie per risolvere il livello"""
        missing_rules = []
        
        # Controlla se manca YOU
        if not rule_analysis['you_rules']:
            # Cerca una regola YOU possibile
            possible_you_rules = [rule for rule in get_all_possible_rules(state) if '-is-you' in rule]
            for rule in possible_you_rules:
                if can_form_rule(state, rule):
                    missing_rules.append(rule)
                    break
        
        # Controlla se manca WIN
        if not rule_analysis['win_rules']:
            possible_win_rules = [rule for rule in get_all_possible_rules(state) if '-is-win' in rule]
            for rule in possible_win_rules:
                if can_form_rule(state, rule):
                    missing_rules.append(rule)
                    break
        
        # Identifica regole strategiche mancanti (come PUSH per sbloccare percorsi)
        strategic_rules = self._identify_strategic_rules(state)
        for rule in strategic_rules:
            if can_form_rule(state, rule) and rule not in state.rules:
                missing_rules.append(rule)
        
        return missing_rules
    
    def _identify_strategic_rules(self, state: GameState) -> List[str]:
        """Identifica regole strategiche che potrebbero essere necessarie"""
        strategic_rules = []
        
        # Cerca oggetti che potrebbero aver bisogno di essere PUSH
        for obj in state.phys:
            if obj.name not in ['baba', 'keke']:  # Esclude giocatori comuni
                push_rule = f"{obj.name}-is-push"
                if push_rule not in state.rules:
                    strategic_rules.append(push_rule)
        
        # Regole comuni utili
        common_strategic = ['rock-is-push', 'wall-is-stop', 'flag-is-win']
        for rule in common_strategic:
            if rule not in state.rules:
                strategic_rules.append(rule)
        
        return strategic_rules
    
    def _solve_with_subgoals(self, initial_state: GameState, target_rules: List[str]) -> Optional[List[Direction]]:
        """Risolve il problema formando prima le regole necessarie"""
        current_state = initial_state.copy()
        total_actions = []
        
        print(f"🎯 Iniziando risoluzione con {len(target_rules)} sotto-obiettivi")
        
        for i, rule in enumerate(target_rules):
            print(f"📌 Sotto-obiettivo {i+1}/{len(target_rules)}: Formando regola '{rule}'")
            
            # Cerca di formare la regola specifica
            actions = self._form_specific_rule(current_state, rule)
            
            if actions is None:
                print(f"❌ Impossibile formare la regola '{rule}'")
                continue
            
            # Applica le azioni per formare la regola
            for action in actions:
                current_state = advance_game_state(action, current_state)
            
            total_actions.extend(actions)
            print(f"✅ Regola '{rule}' formata con {len(actions)} mosse")
            
            # Verifica se il livello è già risolto
            if check_win(current_state):
                print("🎉 Livello risolto durante la formazione delle regole!")
                return total_actions
        
        # Ora risolvi il problema principale
        print("🔄 Avvio ricerca finale...")
        final_actions = self._basic_search(current_state)
        
        if final_actions:
            total_actions.extend(final_actions)
            print(f"🎉 Soluzione trovata! Totale mosse: {len(total_actions)}")
            return total_actions
        else:
            print("❌ Impossibile risolvere dopo la formazione delle regole")
            return None
    
    def _form_specific_rule(self, state: GameState, target_rule: str) -> Optional[List[Direction]]:
        """Cerca di formare una regola specifica usando A*"""
        print(f"🔧 Tentativo di formazione regola: {target_rule}")
        
        if target_rule in state.rules:
            print("✅ Regola già presente!")
            return []
        
        # Ricerca A* specializzata per la formazione di regole
        start_time = time.time()
        time_limit = 60  # secondi per sotto-obiettivo
        
        open_set = [HeapQEntry(0, 0, next(self.counter), [], state)]
        closed_set = {}
        
        for _ in range(self.max_iterations):
            if time.time() - start_time > time_limit:
                print("⏰ Timeout nella formazione regola")
                return None
            
            if not open_set:
                break
            
            current_entry = heapq.heappop(open_set)
            g_score, actions, current_state = current_entry.g_score, current_entry.actions, current_entry.state
            
            # Controlla se la regola è stata formata
            if target_rule in current_state.rules:
                print(f"✅ Regola '{target_rule}' formata con {len(actions)} mosse")
                return actions
            
            # Esplora mosse successive
            for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                if len(actions) >= self.subgoal_max_path:
                    continue
                
                next_state = advance_game_state(direction, current_state.copy())
                new_g_score = g_score + 1
                state_hash = self._get_state_hash(next_state)
                
                if state_hash and new_g_score >= closed_set.get(state_hash, float('inf')):
                    continue
                
                # Euristica specifica per la formazione di regole
                h_score = self._rule_formation_heuristic(next_state, target_rule)
                if h_score == float('inf'):
                    continue
                
                if state_hash:
                    closed_set[state_hash] = new_g_score
                
                f_score = new_g_score + h_score
                heapq.heappush(open_set, HeapQEntry(f_score, new_g_score, next(self.counter), actions + [direction], next_state))
        
        print(f"❌ Impossibile formare la regola '{target_rule}'")
        return None
    
    def _rule_formation_heuristic(self, state: GameState, target_rule: str) -> float:
        """Euristica specifica per la formazione di una regola"""
        if target_rule in state.rules:
            return 0
        
        parts = target_rule.split('-is-')
        if len(parts) != 2:
            return float('inf')
        
        noun_name, prop_name = parts[0], parts[1]
        
        # Trova le parole necessarie
        noun_words = find_word_objects(state, noun_name)
        is_words = find_word_objects(state, 'is')
        prop_words = find_word_objects(state, prop_name)
        
        if not noun_words or not is_words or not prop_words:
            return float('inf')
        
        # Calcola la distanza minima per allineare le parole
        min_formation_cost = float('inf')
        
        for noun in noun_words:
            for is_word in is_words:
                for prop in prop_words:
                    # Controlla allineamento orizzontale
                    if noun.y == is_word.y == prop.y:
                        if noun.x < is_word.x < prop.x:
                            cost = abs(noun.x - is_word.x) + abs(is_word.x - prop.x)
                            min_formation_cost = min(min_formation_cost, cost)
                    
                    # Controlla allineamento verticale
                    if noun.x == is_word.x == prop.x:
                        if noun.y < is_word.y < prop.y:
                            cost = abs(noun.y - is_word.y) + abs(is_word.y - prop.y)
                            min_formation_cost = min(min_formation_cost, cost)
                    
                    # Costo per spostare le parole in posizione
                    align_cost = (manhattan_distance((noun.x, noun.y), (is_word.x, is_word.y)) + 
                                 manhattan_distance((is_word.x, is_word.y), (prop.x, prop.y)))
                    min_formation_cost = min(min_formation_cost, align_cost)
        
        # Aggiungi costo per raggiungere le parole con il giocatore
        if state.players and min_formation_cost != float('inf'):
            player_cost = min([manhattan_distance((p.x, p.y), (w.x, w.y)) 
                              for p in state.players for w in noun_words + is_words + prop_words])
            return min_formation_cost + player_cost
        
        return min_formation_cost if min_formation_cost != float('inf') else 1000
    
    def _basic_search(self, initial_state: GameState) -> Optional[List[Direction]]:
        """Ricerca A* standard una volta che le regole sono pronte"""
        start_time = time.time()
        time_limit = 180  # secondi
        
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
                print("⏰ Timeout nella ricerca finale")
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
        
        return None
    
    def _heuristic(self, state: GameState) -> float:
        """Euristica per la ricerca principale"""
        if check_win(state):
            return 0
        
        rule_analysis = analyze_current_rules(state)
        has_you = bool(rule_analysis['you_rules'])
        
        if not state.players or not has_you:
            cost = self._estimate_rule_formation_cost(state, 'you')
            if cost == float('inf'):
                return float('inf')
            return self.P_NO_YOU_RULE + cost
        
        has_win = bool(rule_analysis['win_rules'])
        
        if has_win:
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
        else:
            cost = self._estimate_rule_formation_cost(state, 'win')
            if cost == float('inf'):
                return float('inf')
            return self.P_NO_WIN_RULE + cost
    
    def _estimate_rule_formation_cost(self, state: GameState, target_property_name: str) -> float:
        """Stima il costo per formare una regola"""
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
        """Genera un hash univoco per lo stato del gioco"""
        components = []
        
        player_pos = sorted([(p.x, p.y) for p in state.players])
        word_pos = sorted([(w.name, w.x, w.y) for w in state.words])
        phys_pos = sorted([(o.name, o.x, o.y) for o in state.phys])
        
        components.append(f"P:{player_pos}")
        components.append(f"W:{word_pos}")
        components.append(f"O:{phys_pos}")
        
        return "|".join(components)

# Alias per compatibilità
class LVL8_A_STARAgent(HIERARCHICALAgent):
    pass