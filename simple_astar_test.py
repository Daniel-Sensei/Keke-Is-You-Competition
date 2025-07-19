"""
Test semplificato per confrontare le performance degli agenti A*.
"""

import time
import json
from typing import Dict

from agents.a_star_AGENT import A_STARAgent
from agents.optimized_astar_v2_AGENT import OPTIMIZED_ASTAR_V2Agent
from baba import make_level, parse_map

def test_single_level(level_data: str, agent_name: str, agent):
    """Testa un agente su un singolo livello."""
    print(f"\n🧪 Test {agent_name}")
    
    start_time = time.time()
    try:
        # Parsa la mappa ASCII e crea il livello
        game_map = parse_map(level_data)
        initial_state = make_level(game_map)
        
        # Esegui l'agente
        actions = agent.search(initial_state, iterations=50000)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if actions:
            print(f"✅ Successo! Tempo: {execution_time:.2f}s, Mosse: {len(actions)}")
            return {'success': True, 'time': execution_time, 'moves': len(actions)}
        else:
            print(f"❌ Fallito. Tempo: {execution_time:.2f}s")
            return {'success': False, 'time': execution_time, 'moves': 0}
            
    except Exception as e:
        end_time = time.time()
        print(f"💥 Errore: {str(e)}")
        return {'success': False, 'time': end_time - start_time, 'moves': 0, 'error': str(e)}

def main():
    """Test principale."""
    print("🚀 Test comparativo A* Agent vs Optimized A* V2 Agent")
    print("=" * 50)
    
    # Carica un livello dal file test
    try:
        with open('json_levels/test_LEVELS.json', 'r') as f:
            data = json.load(f)
        
        # Usa un livello complesso (indice -3, tra gli ultimi)
        level = data['levels'][-3]
        level_ascii = level['ascii']
        level_id = level['id']
        
        print(f"📋 Test su livello ID: {level_id}")
        print(f"🎯 Soluzione attesa: {level.get('solution', 'N/A')}")
        
        # Inizializza gli agenti
        original_agent = A_STARAgent()
        optimized_agent = OPTIMIZED_ASTAR_V2Agent()
        
        # Test agente originale
        original_result = test_single_level(level_ascii, "A* Original", original_agent)
        
        # Pulisci cache se disponibile
        if hasattr(optimized_agent, 'clear_caches'):
            optimized_agent.clear_caches()
        
        # Test agente ottimizzato
        optimized_result = test_single_level(level_ascii, "A* Optimized V2", optimized_agent)
        
        # Confronto risultati
        print(f"\n{'='*50}")
        print("📊 RISULTATI CONFRONTO")
        print(f"{'='*50}")
        
        print(f"🔴 Originale: {'✅' if original_result['success'] else '❌'} - Tempo: {original_result['time']:.2f}s - Mosse: {original_result['moves']}")
        print(f"🔵 Ottimizzato: {'✅' if optimized_result['success'] else '❌'} - Tempo: {optimized_result['time']:.2f}s - Mosse: {optimized_result['moves']}")
        
        if original_result['success'] and optimized_result['success']:
            time_improvement = ((original_result['time'] - optimized_result['time']) / original_result['time']) * 100
            moves_diff = optimized_result['moves'] - original_result['moves']
            
            print(f"\n🎉 MIGLIORAMENTI:")
            print(f"⏱️ Tempo: {time_improvement:+.1f}%")
            print(f"🎮 Mosse: {moves_diff:+d}")
            
            if time_improvement > 0:
                print("✨ L'agente ottimizzato è più veloce!")
            elif time_improvement < -10:
                print("⚠️ L'agente ottimizzato è più lento del 10%+")
            else:
                print("📊 Performance simili")
        
    except Exception as e:
        print(f"❌ Errore nel caricamento del livello: {e}")

if __name__ == "__main__":
    main()
