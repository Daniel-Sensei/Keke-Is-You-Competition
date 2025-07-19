# train_rl.py (Versione Migliorata con Curriculum Learning Avanzato)

import json
import torch
import numpy as np
import os
import time
from collections import defaultdict
from baba_env import BabaEnv
from agents.rl_AGENT import RLAgent  # Importa direttamente, non da agents/

def load_and_categorize_levels(filepaths: list) -> dict:
    """Carica e classifica i livelli per difficoltà basata sulla lunghezza della soluzione."""
    categorized_levels = {
        'tutorial': [],     # <= 10 mosse
        'easy': [],         # 11-20 mosse
        'medium': [],       # 21-40 mosse
        'hard': [],         # 41-60 mosse
        'expert': []        # > 60 mosse
    }
    
    level_metadata = []  # Per tracciare info sui livelli
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Gestisci diversi formati di file JSON
                if isinstance(data, dict) and 'levels' in data:
                    levels = data['levels']
                elif isinstance(data, list):
                    levels = data
                else:
                    print(f"Formato non riconosciuto in {filepath}")
                    continue
                
                for level in levels:
                    ascii_map = level.get('ascii', '')
                    solution = level.get('solution', '')
                    solution_length = len(solution)
                    
                    # Categorizza per difficoltà
                    if solution_length <= 10:
                        categorized_levels['tutorial'].append(ascii_map)
                        category = 'tutorial'
                    elif solution_length <= 20:
                        categorized_levels['easy'].append(ascii_map)
                        category = 'easy'
                    elif solution_length <= 40:
                        categorized_levels['medium'].append(ascii_map)
                        category = 'medium'
                    elif solution_length <= 60:
                        categorized_levels['hard'].append(ascii_map)
                        category = 'hard'
                    else:
                        categorized_levels['expert'].append(ascii_map)
                        category = 'expert'
                    
                    # Salva metadata
                    level_metadata.append({
                        'id': level.get('id', 'unknown'),
                        'name': level.get('name', 'unnamed'),
                        'author': level.get('author', 'unknown'),
                        'solution_length': solution_length,
                        'category': category
                    })
                
            print(f"✅ Caricati {len(levels)} livelli da {filepath}")
            
        except Exception as e:
            print(f"❌ Errore nel caricare {filepath}: {e}")
    
    # Stampa statistiche
    print("\n📊 Statistiche livelli caricati:")
    total = 0
    for category, levels in categorized_levels.items():
        count = len(levels)
        total += count
        print(f"  - {category.capitalize()}: {count} livelli")
    print(f"  - TOTALE: {total} livelli\n")
    
    return categorized_levels, level_metadata

def create_curriculum_schedule(total_episodes: int) -> list:
    """Crea un programma di curriculum learning."""
    schedule = []
    
    # Fase 1: Solo tutorial (10%)
    phase1_episodes = int(total_episodes * 0.1)
    schedule.append({
        'episodes': phase1_episodes,
        'levels': ['tutorial'],
        'name': 'Tutorial Phase'
    })
    
    # Fase 2: Tutorial + Easy (15%)
    phase2_episodes = int(total_episodes * 0.15)
    schedule.append({
        'episodes': phase2_episodes,
        'levels': ['tutorial', 'easy'],
        'name': 'Easy Phase'
    })
    
    # Fase 3: Easy + Medium (25%)
    phase3_episodes = int(total_episodes * 0.25)
    schedule.append({
        'episodes': phase3_episodes,
        'levels': ['easy', 'medium'],
        'name': 'Medium Phase'
    })
    
    # Fase 4: Medium + Hard (25%)
    phase4_episodes = int(total_episodes * 0.25)
    schedule.append({
        'episodes': phase4_episodes,
        'levels': ['medium', 'hard'],
        'name': 'Hard Phase'
    })
    
    # Fase 5: Tutti i livelli (25% rimanente)
    remaining_episodes = total_episodes - sum(p['episodes'] for p in schedule)
    schedule.append({
        'episodes': remaining_episodes,
        'levels': ['tutorial', 'easy', 'medium', 'hard', 'expert'],
        'name': 'All Levels Phase'
    })
    
    return schedule

def validate_agent(agent, validation_levels, max_steps=200):
    """Valida l'agente su un set di livelli di test."""
    results = defaultdict(lambda: {'solved': 0, 'total': 0})
    
    for category, levels in validation_levels.items():
        if not levels:
            continue
            
        # Testa su massimo 5 livelli per categoria
        test_levels = levels[:5] if len(levels) > 5 else levels
        
        for level_ascii in test_levels:
            env = BabaEnv([level_ascii], agent.MODEL_MAX_H, agent.MODEL_MAX_W)
            state, _ = env.reset()
            
            solved = False
            for step in range(max_steps):
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), device=agent.device, 
                                              dtype=torch.float32).unsqueeze(0)
                    action = agent.policy_net(state_tensor).max(1)[1].item()
                
                state, reward, terminated, truncated, info = env.step(action)
                
                if info.get('won', False):
                    solved = True
                    break
                if terminated or truncated:
                    break
            
            results[category]['total'] += 1
            if solved:
                results[category]['solved'] += 1
    
    return results

def print_validation_results(results):
    """Stampa i risultati della validazione."""
    print("\n📈 Risultati validazione:")
    total_solved = 0
    total_tested = 0
    
    for category in ['tutorial', 'easy', 'medium', 'hard', 'expert']:
        if category in results:
            solved = results[category]['solved']
            total = results[category]['total']
            percentage = (solved / total * 100) if total > 0 else 0
            print(f"  - {category.capitalize()}: {solved}/{total} ({percentage:.1f}%)")
            total_solved += solved
            total_tested += total
    
    overall_percentage = (total_solved / total_tested * 100) if total_tested > 0 else 0
    print(f"  - TOTALE: {total_solved}/{total_tested} ({overall_percentage:.1f}%)\n")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    LEVEL_FILES = [
        'json_levels/demo_LEVELS.json',
        'json_levels/full_biy_LEVELS.json'
    ]
    
    NUM_EPISODES_TOTAL = 50000  # Aumentato significativamente
    SAVE_CHECKPOINT_EVERY = 5000
    VALIDATE_EVERY = 2500
    MODEL_SAVE_PATH = 'dqn_baba_improved_model.pth'
    
    print("🚀 === Avvio Training Avanzato per Baba is You === 🚀\n")
    
    # 1. Carica e classifica tutti i livelli
    print("📁 Caricamento livelli...")
    levels_by_difficulty, metadata = load_and_categorize_levels(LEVEL_FILES)
    
    # Verifica che ci siano livelli
    all_maps = []
    for levels in levels_by_difficulty.values():
        all_maps.extend(levels)
    
    if not all_maps:
        print("❌ Nessun livello caricato. Interruzione.")
        exit()
    
    # Calcola dimensioni massime
    max_h = max(len(m.split('\n')) for m in all_maps)
    max_w = max(len(line) for m in all_maps for line in m.split('\n'))
    print(f"📐 Dimensioni massime rilevate: {max_h}x{max_w}")
    
    # 2. Crea schedule del curriculum
    curriculum_schedule = create_curriculum_schedule(NUM_EPISODES_TOTAL)
    print("\n📚 Curriculum Learning Schedule:")
    for i, phase in enumerate(curriculum_schedule):
        print(f"  Fase {i+1} - {phase['name']}: {phase['episodes']} episodi")
    
    # 3. Inizializza ambiente e agente
    print("\n🤖 Inizializzazione agente...")
    
    # Inizia con livelli tutorial
    initial_levels = levels_by_difficulty['tutorial'] if levels_by_difficulty['tutorial'] else all_maps[:10]
    env = BabaEnv(list_of_ascii_maps=initial_levels, max_height=max_h, max_width=max_w)
    agent = RLAgent(env)
    
    # 4. Training con curriculum learning
    print("\n🎮 Inizio training...\n")
    start_time = time.time()
    total_episodes_done = 0
    
    for phase_idx, phase in enumerate(curriculum_schedule):
        print(f"\n{'='*60}")
        print(f"📖 FASE {phase_idx + 1}: {phase['name']}")
        print(f"{'='*60}\n")
        
        # Costruisci il pool di livelli per questa fase
        phase_levels = []
        for level_type in phase['levels']:
            phase_levels.extend(levels_by_difficulty[level_type])
        
        if not phase_levels:
            print(f"⚠️  Nessun livello disponibile per la fase {phase['name']}, skip...")
            continue
        
        print(f"📊 Pool di training: {len(phase_levels)} livelli")
        
        # Aggiorna l'ambiente con i nuovi livelli
        agent.env.list_of_ascii_maps = phase_levels
        
        # Training per questa fase
        agent.train(
            num_episodes=phase['episodes'],
            save_checkpoint_every=SAVE_CHECKPOINT_EVERY
        )
        
        total_episodes_done += phase['episodes']
        
        # Validazione periodica
        if total_episodes_done % VALIDATE_EVERY == 0:
            print("\n🔍 Esecuzione validazione...")
            validation_results = validate_agent(agent, levels_by_difficulty)
            print_validation_results(validation_results)
    
    # 5. Validazione finale
    print("\n🏁 Training completato! Esecuzione validazione finale...")
    final_results = validate_agent(agent, levels_by_difficulty)
    print_validation_results(final_results)
    
    # 6. Salva il modello finale
    print(f"\n💾 Salvataggio modello finale in '{MODEL_SAVE_PATH}'...")
    agent._save_checkpoint(NUM_EPISODES_TOTAL, 0, is_best=True)
    
    # Statistiche finali
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n⏱️  Tempo totale di training: {hours}h {minutes}m {seconds}s")
    print(f"📊 Episodi totali: {NUM_EPISODES_TOTAL}")
    print(f"🎯 Training completato con successo!")
    
    # Salva report finale
    report = {
        'training_time': elapsed_time,
        'total_episodes': NUM_EPISODES_TOTAL,
        'curriculum_schedule': curriculum_schedule,
        'final_validation_results': dict(final_results),
        'level_metadata': metadata
    }
    
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📄 Report di training salvato in 'training_report.json'")
    print("\n✨ Tutto completato! Il modello è pronto per l'uso. ✨")