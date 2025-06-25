"""
Modulo per l'esecuzione degli agenti su livelli di gioco.
Gestisce il caricamento degli agenti, l'esecuzione dei livelli e la raccolta dei risultati.
"""

import importlib.util
import json
import time
import sys
import os
from typing import List, Dict, Any, Union

# Importa moduli necessari da baba.py
from baba import GameState, Direction, make_level, advance_game_state, check_win, parse_map


class Execution:
    def __init__(self, agent_path, level_path, iter_cap=1000, use_cache=False):
        """
        Inizializza un'istanza di esecuzione.
        
        Args:
            agent_path (str): Percorso del file dell'agente
            level_path (str): Percorso del file dei livelli
            iter_cap (int): Limite massimo di iterazioni per livello
            use_cache (bool): Se True, usa la cache per i risultati esistenti
        """
        self.agent_path = agent_path
        self.level_path = level_path
        self.iter_cap = iter_cap
        self.agent = None
        self.levels = []
        self.use_cache = use_cache
        
        print(f"[DEBUG] Inizializzazione esecuzione con agente: {agent_path}")
        print(f"[DEBUG] Set di livelli: {level_path}")
        print(f"[DEBUG] Limite iterazioni: {iter_cap}")
        print(f"[DEBUG] Uso cache: {use_cache}")
        
        # Crea la directory results se non esiste
        self.results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Carica l'agente e i livelli
        self.load_agent(agent_path)
        self.load_levels(level_path)
    
    def load_agent(self, agent_path):
        """
        Carica dinamicamente l'agente dal file specificato.
        
        Args:
            agent_path (str): Percorso del file dell'agente
            
        Returns:
            bool: True se il caricamento è avvenuto con successo
        """
        try:
            print(f"[DEBUG] Tentativo di caricamento dell'agente da: {agent_path}")
            # Determina il nome del modulo dal percorso del file
            module_name = os.path.basename(agent_path).replace('_AGENT.py', '')
            module_name = module_name.upper()+'Agent'
            print(f"[DEBUG] Nome del modulo dell'agente: {module_name}")
            
            # Importa il modulo dinamicamente
            spec = importlib.util.spec_from_file_location(module_name, agent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Determina la classe dell'agente (assumendo che sia nominata come il modulo)
            print(f"[DEBUG] Cerco la classe {module_name} nel modulo")
            agent_class = getattr(module, module_name)
            
            # Istanzia l'agente
            print(f"[DEBUG] Creo un'istanza dell'agente {module_name}")
            self.agent = agent_class()
            
            # Verifica che l'agente abbia il metodo search
            if not hasattr(self.agent, 'search'):
                print(f"ATTENZIONE: L'agente {module_name} non ha un metodo 'search'")
                return False
            
            print(f"[DEBUG] Metodo search trovato nell'agente")
            print(f"Agente {module_name} caricato con successo")
            return True
        except Exception as e:
            print(f"[ERROR] Errore durante il caricamento dell'agente: {e}")
            import traceback
            print(f"[DEBUG] Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def load_levels(self, level_path):
        """
        Carica i livelli dal file JSON.
        
        Args:
            level_path (str): Percorso del file dei livelli
            
        Returns:
            bool: True se il caricamento è avvenuto con successo
        """
        try:
            print(f"[DEBUG] Tentativo di caricamento dei livelli da: {level_path}")
            with open(level_path, 'r') as f:
                level_data = json.load(f)
                
            # Verifica se il formato è quello atteso
            if "levels" in level_data:
                print(f"[DEBUG] Formato JSON con attributo 'levels' trovato")
                self.levels = level_data["levels"]
            else:
                print(f"[DEBUG] Formato JSON senza attributo 'levels', usando l'intero contenuto")
                self.levels = level_data
            
            print(f"[DEBUG] Struttura del primo livello: {self.levels[0].keys() if self.levels else 'nessun livello'}")
            print(f"Caricati {len(self.levels)} livelli da {level_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Errore durante il caricamento dei livelli: {e}")
            import traceback
            print(f"[DEBUG] Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def run_single_level(self, level_data):
        """
        Esegue l'agente su un singolo livello.
        
        Args:
            level_data (dict): Dati del livello da eseguire
            
        Returns:
            dict: Risultati dell'esecuzione (stato, tempo, iterazioni, soluzione, ecc.)
        """
        if not self.agent:
            print(f"[ERROR] Impossibile eseguire il livello: agente non caricato")
            return {'status': 'failed', 'error': 'Agente non caricato'}
        
        # Ottieni la mappa ASCII dal livello
        ascii_map = level_data.get('ascii', '')
        print(f"[DEBUG] Livello: {level_data.get('name', 'sconosciuto')}")
        
        if not ascii_map:
            print(f"[ERROR] Mappa ASCII non trovata nel livello")
            return {'status': 'failed', 'error': 'Mappa ASCII non trovata nel livello'}
        
        # Crea l'ambiente di gioco
        map_data = parse_map(ascii_map)
        game_state = make_level(map_data)
        
        # Esegui l'agente e misura il tempo
        print(f"[DEBUG] Avvio dell'esecuzione dell'agente sul livello")
        start_time = time.time()
        try:
            print(f"[DEBUG] Chiamata al metodo search dell'agente con iterations={self.iter_cap}")
            solution = self.agent.search(game_state, iterations=self.iter_cap)
            elapsed_time = time.time() - start_time
            print(f"[DEBUG] Agente terminato in {elapsed_time:.4f} secondi")
            print(f"[DEBUG] Soluzione trovata: {solution if solution else 'Nessuna'}")
            print(f"[DEBUG] tipo di soluzione: {type(solution)}")
            print(f"[DEBUG] Tipo di elemento nella soluzione: {type(solution[0]) if solution else 'Nessuna'}")
            
            # Converti la soluzione da Direction a stringhe per la serializzazione JSON
            if solution:
                # Controllo se gli elementi sono Direction invece che stringhe
                if hasattr(solution[0], 'name'):  # Se è un enum, avrà l'attributo name
                    # Mappa inversa per la conversione da Direction a stringhe
                    inv_direction_map = {
                        Direction.Left: 'l',
                        Direction.Right: 'r',
                        Direction.Up: 'u',
                        Direction.Down: 'd',
                        Direction.Wait: 's',
                        Direction.Undefined: 'x'
                    }
                    solution = [inv_direction_map.get(action, 'x') for action in solution]
                    print(f"[DEBUG] Soluzione convertita per JSON: {solution}")
            
        except Exception as e:
            print(f"[ERROR] Errore durante l'esecuzione dell'agente: {e}")
            import traceback
            print(f"[DEBUG] Traceback completo:\n{traceback.format_exc()}")
            return {
                'id': str(level_data.get('id', level_data.get('name', 'unknown'))),
                'status': 'error',
                'error': str(e),
                'time': time.time() - start_time,
                'iterations': 0,
                'solution': None,
                'won_level': False,
                'ascii_map': ascii_map
            }
        
        # Verifica la validità della soluzione
        if solution is None:
            return {
                'id': str(level_data.get('id', level_data.get('name', 'unknown'))),
                'status': 'unsolved',
                'time': elapsed_time,
                'iterations': self.iter_cap,  # Assumiamo che l'agente abbia raggiunto il limite massimo
                'solution': None,
                'won_level': False,
                'ascii_map': ascii_map
            }
        
        # Applica la soluzione all'ambiente e verifica se ha vinto
        game_state_copy = game_state.copy()
        success = self._apply_solution(game_state_copy, solution)
        
        # Calcola la lunghezza della soluzione
        solution_length = len(solution) if solution else 0
        
        # Calcola il punteggio di efficienza (time^-1 / solution_length)
        # Un valore più alto indica una soluzione migliore (più veloce e/o più breve)
        efficiency_score = 0
        if success and solution_length > 0 and elapsed_time > 0:
            efficiency_score = (1.0 / elapsed_time) / solution_length
        
        return {
            'id': str(level_data.get('id', level_data.get('name', 'unknown'))),
            'status': 'solved' if success else 'failed',
            'won_level': success,
            'time': elapsed_time,
            'iterations': len(solution) if solution else 0,
            'solution': solution,
            'solution_length': solution_length,
            'efficiency_score': efficiency_score,
            'ascii_map': ascii_map
        }
    
    def _apply_solution(self, game_state: GameState, solution: List[str]) -> bool:
        """
        Applica una soluzione a un ambiente di gioco e verifica se ha portato alla vittoria.
        
        Args:
            game_state (GameState): Lo stato di gioco iniziale
            solution (List[str]): Lista di azioni (l, r, u, d)
            
        Returns:
            bool: True se la soluzione porta alla vittoria, False altrimenti
        """
        if not solution:
            print("[DEBUG] Soluzione vuota, impossibile applicarla")
            return False
        
        print(f"[DEBUG] Verifica della soluzione: {solution}")
        
        # Mappa le stringhe di direzione alle enum Direction
        direction_map = {
            'l': Direction.Left,
            'r': Direction.Right,
            'u': Direction.Up,
            'd': Direction.Down,
            's': Direction.Wait,
            'x': Direction.Undefined
        }
        
        # Applica ogni azione della soluzione
        for i, action in enumerate(solution):
            # Converti l'azione in Direction
            direction = direction_map.get(action, Direction.Undefined)
            
            if direction == Direction.Undefined:
                print(f"[ERROR] Azione non valida: {action}")
                return False
                
            # Avanza lo stato di gioco
            game_state = advance_game_state(direction, game_state)
            
            # Verifica se il gioco è stato vinto
            if check_win(game_state):
                return True
        
        # Verifica finale dopo l'applicazione completa
        return check_win(game_state)
    
    def run_all_levels(self, callback=None):
        """
        Esegue l'agente su tutti i livelli caricati.
        Se use_cache è True, controlla prima se esistono risultati nella cache.
        
        Args:
            callback (callable, optional): Funzione di callback per aggiornamenti progressivi
                                         La funzione di callback riceve (current_index, total_levels, current_results)
        
        Returns:
            dict: Report completo con i risultati per ogni livello e metriche riassuntive
        """
        if not self.agent:
            return {'error': 'Agente non caricato'}
        
        agent_id = os.path.basename(self.agent_path).replace('.py', '')
        level_set_id = os.path.basename(self.level_path).replace('.json', '')
        total_levels = len(self.levels)
        
        # Verifica se ci sono risultati in cache e use_cache è True
        if self.use_cache:
            cached_results = self._load_results_from_cache()
            if cached_results:
                print(f"[DEBUG] Trovati risultati in cache per {agent_id} su {level_set_id}")
                
                # Se c'è un callback, simula il progresso per ogni livello
                if callback and callable(callback):
                    levels = cached_results.get('levels', [])
                    # Simula il progresso per ogni livello
                    for i in range(len(levels)):
                        current_report = {
                            'agent': agent_id,
                            'level_set': level_set_id,
                            'levels': levels[:i+1],  # Prende solo i livelli fino all'indice corrente
                            'summary': cached_results.get('summary', {}),
                            'progress': {
                                'current': i + 1,
                                'total': total_levels,
                                'percentage': ((i + 1) / total_levels) * 100
                            },
                            'from_cache': True  # Indica che i risultati provengono dalla cache
                        }
                        callback(i + 1, total_levels, current_report)
                        # Breve pausa per simulare il tempo di esecuzione
                        time.sleep(0.1)
                
                return cached_results
                
        # Se non ci sono risultati in cache o use_cache è False, esegui normalmente
        results = []
        
        for i, level in enumerate(self.levels):
            try:
                print(f"Esecuzione livello {i+1}/{total_levels}: {level.get('name', f'level_{i+1}')}")
                level_result = self.run_single_level(level)
                results.append(level_result)
                print(f"Livello {i+1}/{total_levels} completato: {level_result['status']}")
                
                # Se è stata fornita una funzione di callback, invoca l'aggiornamento progressivo
                if callback and callable(callback):
                    current_report = {
                        'agent': agent_id,
                        'level_set': level_set_id,
                        'levels': results.copy(),  # Copia per evitare problemi di riferimento
                        'summary': self.calculate_summary(results),
                        'progress': {
                            'current': i + 1,
                            'total': total_levels,
                            'percentage': ((i + 1) / total_levels) * 100
                        },
                        'from_cache': False  # Indica che i risultati sono freschi
                    }
                    callback(i + 1, total_levels, current_report)
                    
            except Exception as e:
                print(f"Errore durante l'esecuzione del livello {i+1}: {e}")
                results.append({
                    'id': level.get('name', f'level_{i+1}'),
                    'status': 'error',
                    'error': str(e),
                    'time': 0,
                    'iterations': 0,
                    'solution': None,
                    'won_level': False,
                    'ascii_map': level.get('ascii', '')
                })
                
                # Invia anche gli errori attraverso il callback se disponibile
                if callback and callable(callback):
                    current_report = {
                        'agent': agent_id,
                        'level_set': level_set_id,
                        'levels': results.copy(),
                        'summary': self.calculate_summary(results),
                        'progress': {
                            'current': i + 1,
                            'total': total_levels,
                            'percentage': ((i + 1) / total_levels) * 100
                        },
                        'from_cache': False
                    }
                    callback(i + 1, total_levels, current_report)
        
        final_report = {
            'agent': agent_id,
            'level_set': level_set_id,
            'levels': results,
            'summary': self.calculate_summary(results),
            'progress': {
                'current': total_levels,
                'total': total_levels,
                'percentage': 100
            }
        }
        
        # Salva i risultati nella cache se use_cache è True
        if self.use_cache:
            self._save_results_to_cache(final_report)
        
        return final_report
    
    def calculate_summary(self, level_results):
        """
        Calcola le metriche riassuntive dai risultati dei livelli.
        
        Args:
            level_results (list): Lista dei risultati per ogni livello
            
        Returns:
            dict: Metriche riassuntive (livelli totali, livelli risolti, accuratezza, ecc.)
        """
        total_levels = len(level_results)
        solved_levels = sum(1 for lvl in level_results if lvl.get('status') == 'solved')
        accuracy = solved_levels / total_levels if total_levels > 0 else 0
        
        # Calcola medie
        times = [lvl.get('time', 0) for lvl in level_results]
        iterations = [lvl.get('iterations', 0) for lvl in level_results if lvl.get('iterations') is not None]
        
        # Filtra i livelli risolti per calcolare l'efficienza media
        solved_results = [lvl for lvl in level_results if lvl.get('status') == 'solved']
        efficiency_scores = [lvl.get('efficiency_score', 0) for lvl in solved_results]
        solution_lengths = [lvl.get('solution_length', 0) for lvl in solved_results]
        
        average_time = sum(times) / total_levels if total_levels > 0 else 0
        average_iterations = sum(iterations) / len(iterations) if iterations else 0
        average_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        average_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        
        return {
            'total_levels': total_levels,
            'solved_levels': solved_levels,
            'accuracy': accuracy,
            'average_time': average_time,
            'average_iterations': average_iterations,
            'average_efficiency': average_efficiency,
            'average_solution_length': average_solution_length
        }
    
    def _generate_cache_key(self):
        """
        Genera una chiave univoca per la cache basata su agent_path, level_path e iter_cap.
        
        Returns:
            str: Chiave di cache
        """
        # Estrae solo il nome del file dall'agent_path e level_path
        agent_name = os.path.basename(self.agent_path).replace('.py', '')
        level_name = os.path.basename(self.level_path).replace('.json', '')
        
        # Crea una chiave che combina i tre parametri
        key = f"{agent_name}_{level_name}_{self.iter_cap}"
        return key
    
    def _get_cache_file_path(self):
        """
        Ottiene il percorso del file di cache per la configurazione corrente.
        
        Returns:
            str: Percorso completo del file di cache
        """
        cache_key = self._generate_cache_key()
        return os.path.join(self.results_dir, f"{cache_key}.json")
    
    def _save_results_to_cache(self, results):
        """
        Salva i risultati dell'esecuzione nella cache.
        
        Args:
            results (dict): Risultati dell'esecuzione
            
        Returns:
            bool: True se il salvataggio è avvenuto con successo
        """
        cache_file_path = self._get_cache_file_path()
        try:
            # Aggiungi un timestamp ai risultati
            results_with_timestamp = results.copy()
            results_with_timestamp['cached_at'] = time.time()
            
            with open(cache_file_path, 'w') as f:
                json.dump(results_with_timestamp, f, indent=2)
            
            print(f"[DEBUG] Risultati salvati nella cache: {cache_file_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Errore durante il salvataggio dei risultati nella cache: {e}")
            import traceback
            print(f"[DEBUG] Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def _load_results_from_cache(self):
        """
        Carica i risultati dell'esecuzione dalla cache, se esistono.
        
        Returns:
            dict: Risultati dell'esecuzione o None se la cache non esiste
        """
        cache_file_path = self._get_cache_file_path()
        
        if not os.path.exists(cache_file_path):
            print(f"[DEBUG] Cache non trovata: {cache_file_path}")
            return None
        
        try:
            with open(cache_file_path, 'r') as f:
                cached_results = json.load(f)
            
            print(f"[DEBUG] Risultati caricati dalla cache: {cache_file_path}")
            print(f"[DEBUG] Cache creata il: {time.ctime(cached_results.get('cached_at', 0))}")
            
            # Ritorna i risultati senza il timestamp
            return cached_results
        except Exception as e:
            print(f"[ERROR] Errore durante il caricamento dei risultati dalla cache: {e}")
            import traceback
            print(f"[DEBUG] Traceback completo:\n{traceback.format_exc()}")
            return None