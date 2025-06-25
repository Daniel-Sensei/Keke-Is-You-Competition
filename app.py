from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import os
import json
import queue
import threading
import time
from functools import partial
# Importa i moduli creati
from execution import Execution
# Importa i moduli di baba per la generazione degli stati del gioco
from baba import GameState, Direction, advance_game_state, parse_map, make_level, check_win, double_map_to_string

app = Flask(__name__)

# Configura le directory per i template e i file statici
app.template_folder = os.path.abspath('./templates')
app.static_folder = os.path.abspath('./static')

# Directory dei livelli e degli agenti (adattare i percorsi se necessario)
LEVELS_DIR = os.path.abspath(os.path.join('json_levels'))
AGENTS_DIR = os.path.abspath(os.path.join('agents'))

def discover_files(directory, extension):
    """Scopre file con una data estensione in una directory."""
    if not os.path.exists(directory):
        return []
    files = []
    for f in os.listdir(directory):
        if f.endswith(extension) and not f.startswith('.'): # Ignora file nascosti
            files.append({'id': f.replace(extension, ''), 'name': f.replace(extension, '')})
    return files

@app.after_request
def add_header(response):
    """Disabilita la cache per tutte le risposte HTTP."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    """Renderizza la pagina principale."""
    return render_template('index.html')

@app.route('/get_agents')
def get_agents():
    """Restituisce l'elenco degli agenti disponibili."""
    agents = discover_files(AGENTS_DIR, '_AGENT.py')
    # Rimuovi l'estensione '_AGENT.py' e '_AGENT' per l'ID e il nome
    for agent in agents:
        agent['id'] = agent['id'].replace('_AGENT', '')
        agent['name'] = agent['name'].replace('_AGENT', '')
    return jsonify(agents)

@app.route('/get_level_sets')
def get_level_sets():
    """Restituisce l'elenco dei set di livelli disponibili."""
    level_sets = discover_files(LEVELS_DIR, '_LEVELS.json')
    # Rimuovi l'estensione '_LEVELS.json' e '_LEVELS' per l'ID e il nome
    for ls in level_sets:
        ls['id'] = ls['id'].replace('_LEVELS', '')
        ls['name'] = ls['name'].replace('_LEVELS', '')
    return jsonify(level_sets)

# Dizionario globale per memorizzare code di eventi per ogni session_id
progress_queues = {}

# --- Endpoint per avviare una simulazione ed ottenere un session_id ---
@app.route('/solve_level_set', methods=['POST'])
def solve_level_set():
    data = request.get_json()
    agent_id = data.get('agent')
    level_set_id = data.get('levelSet')
    iterations = data.get('iterations', 1000) # Valore di default per le iterazioni
    use_cache = data.get('useCache', False)   # Nuovo parametro per utilizzare la cache
    
    if not agent_id or not level_set_id:
        return jsonify({'error': 'Agente o set di livelli mancante'}), 400

    agent_file = os.path.join(AGENTS_DIR, f"{agent_id}_AGENT.py")
    level_file = os.path.join(LEVELS_DIR, f"{level_set_id}_LEVELS.json")

    if not os.path.exists(agent_file):
        return jsonify({'error': f'File agente non trovato: {agent_file}'}), 404
    if not os.path.exists(level_file):
        return jsonify({'error': f'File set di livelli non trovato: {level_file}'}), 404

    # Crea un ID univoco per questa sessione
    import uuid
    session_id = str(uuid.uuid4())
    
    # Crea una nuova coda per gli aggiornamenti di questa sessione
    progress_queue = queue.Queue()
    progress_queues[session_id] = progress_queue
    
    print(f"[DEBUG] Richiesta di esecuzione con use_cache={use_cache}")
    
    def progress_callback(current, total, report):
        """Callback che mette gli aggiornamenti di progresso nella coda"""
        progress_data = {
            'current': current,
            'total': total,
            'report': report
        }
        progress_queue.put(progress_data)
    
    try:
        # Avvia la simulazione in un thread separato
        def run_execution():
            try:
                # Inizializza l'executor
                print(f"Avvio simulazione: Agente={agent_id}, Set Livelli={level_set_id}, Iterazioni={iterations}, Uso Cache={use_cache}")
                execution_instance = Execution(agent_path=agent_file, level_path=level_file, iter_cap=iterations, use_cache=use_cache)
                
                # Esegui tutti i livelli con callback per aggiornamenti progressivi
                report = execution_instance.run_all_levels(callback=progress_callback)
                
                # Segnala il completamento mettendo None nella coda
                progress_queue.put(None)
                
                # Debug: stampa il report finale
                print(f"[DEBUG] Report generato completamente:")
                print(f"[DEBUG] - Agent: {report.get('agent')}")
                print(f"[DEBUG] - Level set: {report.get('level_set')}")
                print(f"[DEBUG] - Livelli totali: {report.get('summary', {}).get('total_levels')}")
                print(f"[DEBUG] - Livelli risolti: {report.get('summary', {}).get('solved_levels')}")
                
                # Pulisci la coda dopo un po' di tempo (es. 10 minuti)
                def cleanup_queue():
                    time.sleep(600)  # 10 minuti
                    if session_id in progress_queues:
                        del progress_queues[session_id]
                        print(f"[DEBUG] Pulita coda per session_id: {session_id}")
                
                cleanup_thread = threading.Thread(target=cleanup_queue)
                cleanup_thread.daemon = True
                cleanup_thread.start()
                
            except Exception as e:
                import traceback
                print(f"Errore durante l'esecuzione della simulazione: {e}")
                print(f"Traceback completo:\n{traceback.format_exc()}")
                # Segnala l'errore
                progress_queue.put({'error': str(e)})
                progress_queue.put(None)  # Segnala completamento
        
        # Avvia il thread di esecuzione
        execution_thread = threading.Thread(target=run_execution)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Restituisci l'ID di sessione al client
        return jsonify({
            'session_id': session_id,
            'status': 'processing'
        })

    except Exception as e:
        import traceback
        print(f"Errore durante l'avvio della simulazione: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        
        # Pulisci la coda in caso di errore
        if session_id in progress_queues:
            del progress_queues[session_id]
            
        return jsonify({'error': str(e)}), 500

# --- Endpoint per ricevere aggiornamenti di progresso tramite SSE ---
@app.route('/progress/<session_id>')
def get_progress(session_id):
    if session_id not in progress_queues:
        return jsonify({'error': 'Session ID non valido o scaduto'}), 404
    
    progress_queue = progress_queues[session_id]
    
    def generate():
        try:
            while True:
                # Attendi un aggiornamento dalla coda
                progress_data = progress_queue.get()
                
                # Se è None, significa che l'esecuzione è terminata
                if progress_data is None:
                    yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                    break
                
                # Se è un errore, segnalalo
                if 'error' in progress_data:
                    yield f"data: {json.dumps({'status': 'error', 'error': progress_data['error']})}\n\n"
                    break
                
                # Altrimenti, invia l'aggiornamento di progresso
                yield f"data: {json.dumps(progress_data)}\n\n"
        except GeneratorExit:
            # Il client si è disconnesso
            print(f"[DEBUG] Client disconnesso da SSE per session_id: {session_id}")
            pass
    
    response = Response(stream_with_context(generate()), 
                       mimetype='text/event-stream')
    
    # Aggiungi header necessari per SSE
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Per Nginx
    response.headers['Connection'] = 'keep-alive'
    
    return response

@app.route('/get_level_details/<level_set_id>/<level_id>')
def get_level_details(level_set_id, level_id):
    level_file_path = os.path.join(LEVELS_DIR, f"{level_set_id}_LEVELS.json")
    if not os.path.exists(level_file_path):
        return jsonify({'error': 'File del set di livelli non trovato'}), 404

    try:
        with open(level_file_path, 'r') as f:
            level_data = json.load(f)
        
        # Assicurati che levels_list sia sempre una lista
        if isinstance(level_data, dict) and 'levels' in level_data:
            levels_list = level_data['levels']
        elif isinstance(level_data, list):
            levels_list = level_data
        else:
            return jsonify({'error': 'Formato livelli non valido'}), 500
        
        target_level = None
        # Cerca il livello per id, name o indice se non corrisponde direttamente
        for i, lvl in enumerate(levels_list):
            if isinstance(lvl, dict):  # Verifica che lvl sia un dizionario
                # Prima controlla l'ID numerico come stringa
                if str(lvl.get('id', '')) == level_id:
                    target_level = lvl
                    break
                # Poi controlla il nome
                if lvl.get('name', '') == level_id:
                    target_level = lvl
                    break
                # Infine controlla per indice
                if f"level_{i+1}" == level_id:
                    target_level = lvl
                    break
        
        if target_level:
            return jsonify({
                'id': str(target_level.get('id', level_id)),
                'ascii_map': target_level.get('ascii', ''),  # Usa 'ascii' invece di 'ascii_map'
                'solution_path': target_level.get('solution_path', '') # Se disponibile
            })
        else:
            return jsonify({'error': 'Livello non trovato'}), 404
            
    except Exception as e:
        import traceback
        print(f"Errore durante il caricamento dei dettagli del livello: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# --- Endpoint per generare gli stati del gioco ---
@app.route('/generate_game_states', methods=['POST'])
def generate_game_states():
    """Genera tutti gli stati del gioco basati su livello e soluzione usando la logica di baba.py"""
    try:
        data = request.get_json()
        ascii_map = data.get('ascii_map')
        solution = data.get('solution', [])
        
        if not ascii_map:
            return jsonify({'error': 'Mappa ASCII mancante'}), 400
        
        # Mappa le stringhe di direzione alle enum Direction (come in execution.py)
        direction_map = {
            'l': Direction.Left,
            'r': Direction.Right,
            'u': Direction.Up,
            'd': Direction.Down,
            's': Direction.Wait,
            'x': Direction.Undefined
        }
        
        # Parsing della mappa ASCII
        game_map = parse_map(ascii_map)
        initial_game_state = make_level(game_map)
        
        # Lista per memorizzare tutti gli stati del gioco
        game_states = []
        
        # Aggiungi lo stato iniziale
        initial_state_str = double_map_to_string(initial_game_state.obj_map, initial_game_state.back_map)
        game_states.append({
            'step': 0,
            'move': 'start',
            'map_string': initial_state_str,
            'ascii_map': ascii_map,
            'won': check_win(initial_game_state)
        })
        
        # Applica ogni mossa della soluzione
        current_game_state = initial_game_state
        for i, move in enumerate(solution):
            # Converti la mossa in Direction
            direction = direction_map.get(move.lower(), Direction.Undefined)
            
            if direction == Direction.Undefined:
                print(f"[WARNING] Mossa non valida ignorata: {move}")
                continue
            
            # Avanza lo stato di gioco
            current_game_state = advance_game_state(direction, current_game_state)
            
            # Converte lo stato in stringa
            state_str = double_map_to_string(current_game_state.obj_map, current_game_state.back_map)
            
            # Aggiungi lo stato alla lista
            game_states.append({
                'step': i + 1,
                'move': move,
                'map_string': state_str,
                'ascii_map': state_str,  # Lo stato corrente in formato ASCII
                'won': check_win(current_game_state)
            })
            
            # Se il gioco è stato vinto, fermati qui
            if check_win(current_game_state):
                break
        
        return jsonify({
            'status': 'success',
            'total_steps': len(game_states),
            'states': game_states
        })
        
    except Exception as e:
        import traceback
        print(f"Errore durante la generazione degli stati del gioco: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# --- Endpoint per gestire la cache dei risultati ---
@app.route('/manage_cache', methods=['POST'])
def manage_cache():
    """Endpoint per gestire la cache dei risultati."""
    data = request.get_json()
    action = data.get('action')
    
    # Percorso della directory dei risultati
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    
    if action == 'clear_all':
        # Elimina tutti i file di cache
        try:
            if os.path.exists(results_dir):
                for file in os.listdir(results_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(results_dir, file))
                return jsonify({'status': 'success', 'message': 'Cache eliminata con successo'})
            return jsonify({'status': 'success', 'message': 'Nessuna cache da eliminare'})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    elif action == 'get_stats':
        # Ottieni statistiche sulla cache
        try:
            stats = {'total_files': 0, 'total_size': 0, 'files': []}
            
            if os.path.exists(results_dir):
                for file in os.listdir(results_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(results_dir, file)
                        file_size = os.path.getsize(file_path)
                        stats['total_files'] += 1
                        stats['total_size'] += file_size
                        
                        # Ottieni informazioni sul file di cache
                        try:
                            with open(file_path, 'r') as f:
                                cache_data = json.load(f)
                            
                            stats['files'].append({
                                'name': file,
                                'size': file_size,
                                'agent': cache_data.get('agent', 'unknown'),
                                'level_set': cache_data.get('level_set', 'unknown'),
                                'cached_at': cache_data.get('cached_at', 0),
                                'summary': cache_data.get('summary', {})
                            })
                        except:
                            stats['files'].append({
                                'name': file,
                                'size': file_size,
                                'error': 'Impossibile leggere il file'
                            })
            
            return jsonify({'status': 'success', 'stats': stats})
        
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    else:
        return jsonify({'status': 'error', 'error': 'Azione non valida'}), 400

# --- Endpoint per salvare le soluzioni in formato JSON ---
@app.route('/save_solution', methods=['POST'])
def save_solution():
    """Salva una soluzione step-by-step in formato JSON."""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data or 'data' not in data:
            return jsonify({'status': 'error', 'error': 'Dati mancanti'}), 400
        
        filename = data['filename']
        solution_data = data['data']
        
        # Crea la directory solutions se non esiste
        solutions_dir = os.path.abspath('./solutions')
        if not os.path.exists(solutions_dir):
            os.makedirs(solutions_dir)
            print(f"Creata directory: {solutions_dir}")
        
        # Salva il file JSON
        filepath = os.path.join(solutions_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(solution_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Soluzione salvata: {filepath}")
        return jsonify({
            'status': 'success', 
            'message': f'Soluzione salvata: {filename}',
            'filepath': filepath
        })
        
    except Exception as e:
        print(f"❌ Errore nel salvataggio: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# --- Endpoint per ottenere la classifica degli agenti ---
@app.route('/get_leaderboard')
def get_leaderboard():
    """Genera una classifica degli agenti basata sui risultati memorizzati nella cache."""
    try:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
        
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'success',
                'leaderboard': [],
                'message': 'Nessun risultato disponibile'
            })
        
        agents_data = {}
        
        # Leggi tutti i file di cache
        for filename in os.listdir(results_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                agent_name = data.get('agent', 'unknown')
                level_set = data.get('level_set', 'unknown')
                
                # Crea una chiave unica per agente + level_set
                key = f"{agent_name}_{level_set}"
                
                if key not in agents_data:
                    agents_data[key] = {
                        'agent': agent_name,
                        'level_set': level_set,
                        'executions': []
                    }
                
                # Calcola le metriche per questa esecuzione
                levels = data.get('levels', [])
                summary = data.get('summary', {})
                
                total_levels = len(levels)
                solved_levels = sum(1 for level in levels if level.get('won_level', False))
                total_time = sum(level.get('time', 0) for level in levels if level.get('won_level', False))
                total_iterations = sum(level.get('iterations', 0) for level in levels if level.get('won_level', False))
                
                accuracy = (solved_levels / total_levels * 100) if total_levels > 0 else 0
                avg_time = (total_time / solved_levels) if solved_levels > 0 else 0
                avg_iterations = (total_iterations / solved_levels) if solved_levels > 0 else 0
                
                # Ottieni l'efficienza media dai dati di sommario se disponibile
                avg_efficiency = summary.get('average_efficiency', 0) if summary else 0
                
                execution_data = {
                    'filename': filename,
                    'cached_at': data.get('cached_at', 0),
                    'total_levels': total_levels,
                    'solved_levels': solved_levels,
                    'accuracy': accuracy,
                    'avg_time': avg_time,
                    'avg_iterations': avg_iterations,
                    'avg_efficiency': avg_efficiency,  # Aggiungiamo l'efficienza media
                    'total_time': total_time,
                    'total_iterations': total_iterations
                }
                
                agents_data[key]['executions'].append(execution_data)
                
            except Exception as e:
                print(f"Errore nel leggere il file {filename}: {e}")
                continue
        
        # Genera la classifica
        leaderboard = []
        for key, agent_data in agents_data.items():
            executions = agent_data['executions']
            if not executions:
                continue
            
            # Prendi la migliore esecuzione (più alta accuratezza, poi minor tempo medio)
            best_execution = max(executions, key=lambda x: (x['accuracy'], -x['avg_time']))
            
            leaderboard_entry = {
                'agent': agent_data['agent'],
                'level_set': agent_data['level_set'],
                'accuracy': best_execution['accuracy'],
                'solved_levels': best_execution['solved_levels'],
                'total_levels': best_execution['total_levels'],
                'avg_time': best_execution['avg_time'],
                'avg_iterations': best_execution['avg_iterations'],
                'avg_efficiency': best_execution.get('avg_efficiency', 0),  # Aggiungiamo l'efficienza media
                'total_executions': len(executions),
                'best_execution_file': best_execution['filename']
            }
            
            leaderboard.append(leaderboard_entry)
        
        # Ordina la classifica per accuratezza (decrescente), poi per tempo medio (crescente)
        leaderboard.sort(key=lambda x: (-x['accuracy'], x['avg_time']))
        
        return jsonify({
            'status': 'success',
            'leaderboard': leaderboard,
            'total_agents': len(leaderboard)
        })
        
    except Exception as e:
        import traceback
        print(f"Errore nella generazione della classifica: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    # Trova una porta libera
    port = 5001
    print(f"Avvio del server Flask su http://localhost:{port}")
    app.run(debug=True, port=port)
