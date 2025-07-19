"""
Test di confronto tra A* Agent originale e la versione ottimizzata.
Misura tempi di esecuzione, nodi esplorati e qualità delle soluzioni.
"""

import time
import json
from typing import Dict, List, Optional
import sys
import os

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.a_star_AGENT import A_STARAgent
from agents.optimized_astar_v2_AGENT import OptimizedAStarV2Agent
from execution import Execution
from baba import make_level

def load_test_levels() -> Dict[int, str]:
    """Carica alcuni livelli di test."""
    test_levels = {}
    level_files = [1, 2, 3, 5, 8, 10, 15, 20]  # Livelli di difficoltà crescente
    
    for level_num in level_files:
        try:
            with open(f'json_levels/level_{level_num}.json', 'r') as f:
                level_data = json.load(f)
                test_levels[level_num] = level_data
        except FileNotFoundError:
            print(f"⚠️ Livello {level_num} non trovato")
            continue
    
    return test_levels

def run_performance_test(agent, agent_name: str, level_num: int, level_data: str, max_time: int = 120) -> Dict:
    """Esegue un test di performance su un singolo livello."""
    print(f"\n🧪 Test {agent_name} su livello {level_num}")
    
    start_time = time.time()
    
    try:
        # Parse del livello
        initial_state = make_level(level_data)
        
        # Esecuzione dell'agente
        actions = agent.search(initial_state, iterations=50000)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        result = {
            'level': level_num,
            'agent': agent_name,
            'success': actions is not None,
            'execution_time': execution_time,
            'actions_count': len(actions) if actions else 0,
            'actions': actions
        }
        
        if actions:
            print(f"✅ Successo! Tempo: {execution_time:.2f}s, Mosse: {len(actions)}")
        else:
            print(f"❌ Fallito. Tempo: {execution_time:.2f}s")
            
        return result
        
    except Exception as e:
        end_time = time.time()
        print(f"💥 Errore: {str(e)}")
        return {
            'level': level_num,
            'agent': agent_name,
            'success': False,
            'execution_time': end_time - start_time,
            'error': str(e),
            'actions_count': 0
        }

def compare_agents():
    """Confronta le performance dei due agenti."""
    print("🚀 Confronto performance A* Agent vs Optimized A* V2 Agent")
    print("=" * 60)
    
    # Carica i livelli di test
    test_levels = load_test_levels()
    if not test_levels:
        print("❌ Nessun livello di test trovato!")
        return
    
    print(f"📋 Livelli caricati: {list(test_levels.keys())}")
    
    # Inizializza gli agenti
    original_agent = A_STARAgent()
    optimized_agent = OptimizedAStarV2Agent()
    
    results = []
    
    # Test su ogni livello
    for level_num, level_data in test_levels.items():
        print(f"\n{'='*40}")
        print(f"📌 LIVELLO {level_num}")
        print(f"{'='*40}")
        
        # Test agente originale
        original_result = run_performance_test(original_agent, "A*_Original", level_num, level_data)
        results.append(original_result)
        
        # Pulisci cache se possibile
        if hasattr(optimized_agent, 'clear_caches'):
            optimized_agent.clear_caches()
        
        # Test agente ottimizzato
        optimized_result = run_performance_test(optimized_agent, "A*_Optimized_V2", level_num, level_data)
        results.append(optimized_result)
        
        # Confronto diretto
        print(f"\n📊 CONFRONTO LIVELLO {level_num}:")
        if original_result['success'] and optimized_result['success']:
            time_improvement = ((original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time']) * 100
            moves_diff = optimized_result['actions_count'] - original_result['actions_count']
            
            print(f"⏱️  Tempo originale: {original_result['execution_time']:.2f}s")
            print(f"⏱️  Tempo ottimizzato: {optimized_result['execution_time']:.2f}s")
            print(f"🎯 Miglioramento tempo: {time_improvement:+.1f}%")
            print(f"🎮 Mosse originale: {original_result['actions_count']}")
            print(f"🎮 Mosse ottimizzato: {optimized_result['actions_count']}")
            print(f"📈 Differenza mosse: {moves_diff:+d}")
        else:
            print(f"⚖️  Originale: {'✅' if original_result['success'] else '❌'}")
            print(f"⚖️  Ottimizzato: {'✅' if optimized_result['success'] else '❌'}")
    
    # Statistiche finali
    print_final_statistics(results)

def print_final_statistics(results: List[Dict]):
    """Stampa le statistiche finali del confronto."""
    print(f"\n{'='*60}")
    print("📈 STATISTICHE FINALI")
    print(f"{'='*60}")
    
    # Separa i risultati per agente
    original_results = [r for r in results if r['agent'] == 'A*_Original']
    optimized_results = [r for r in results if r['agent'] == 'A*_Optimized_V2']
    
    # Successi
    original_successes = sum(1 for r in original_results if r['success'])
    optimized_successes = sum(1 for r in optimized_results if r['success'])
    
    print(f"🎯 Successi originale: {original_successes}/{len(original_results)}")
    print(f"🎯 Successi ottimizzato: {optimized_successes}/{len(optimized_results)}")
    
    # Tempi medi (solo per i successi)
    original_times = [r['execution_time'] for r in original_results if r['success']]
    optimized_times = [r['execution_time'] for r in optimized_results if r['success']]
    
    if original_times and optimized_times:
        avg_original_time = sum(original_times) / len(original_times)
        avg_optimized_time = sum(optimized_times) / len(optimized_times)
        avg_improvement = ((avg_original_time - avg_optimized_time) / avg_original_time) * 100
        
        print(f"⏰ Tempo medio originale: {avg_original_time:.2f}s")
        print(f"⏰ Tempo medio ottimizzato: {avg_optimized_time:.2f}s")
        print(f"🚀 Miglioramento medio: {avg_improvement:+.1f}%")
    
    # Mosse medie
    original_moves = [r['actions_count'] for r in original_results if r['success']]
    optimized_moves = [r['actions_count'] for r in optimized_results if r['success']]
    
    if original_moves and optimized_moves:
        avg_original_moves = sum(original_moves) / len(original_moves)
        avg_optimized_moves = sum(optimized_moves) / len(optimized_moves)
        
        print(f"🎮 Mosse medie originale: {avg_original_moves:.1f}")
        print(f"🎮 Mosse medie ottimizzato: {avg_optimized_moves:.1f}")
        print(f"📊 Differenza mosse: {avg_optimized_moves - avg_original_moves:+.1f}")
    
    # Salva i risultati in un file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"performance_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Risultati salvati in: {filename}")

def quick_test():
    """Test rapido su un singolo livello facile."""
    print("🏃‍♂️ Test rapido su livello 1")
    
    try:
        with open('json_levels/level_1.json', 'r') as f:
            level_data = json.load(f)
        
        original_agent = A_STARAgent()
        optimized_agent = OptimizedAStarV2Agent()
        
        print("\n🔄 Test agente originale...")
        original_result = run_performance_test(original_agent, "A*_Original", 1, level_data)
        
        print("\n🔄 Test agente ottimizzato...")
        optimized_result = run_performance_test(optimized_agent, "A*_Optimized_V2", 1, level_data)
        
        if original_result['success'] and optimized_result['success']:
            improvement = ((original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time']) * 100
            print(f"\n🎉 Miglioramento: {improvement:+.1f}%")
        
    except Exception as e:
        print(f"❌ Errore nel test rapido: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test di confronto performance A* agents")
    parser.add_argument("--quick", action="store_true", help="Esegui solo un test rapido")
    parser.add_argument("--levels", nargs="+", type=int, help="Livelli specifici da testare")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        compare_agents()
