"""
Test di confronto tra tutte le versioni degli agenti A*.
"""

import time
import json
from typing import Dict

from agents.a_star_AGENT import A_STARAgent
from agents.optimized_astar_v2_AGENT import OPTIMIZED_ASTAR_V2Agent
from agents.speed_optimized_astar_AGENT import OptimizedSpeedAgent
from baba import make_level, parse_map

def test_agent(level_data: str, agent_name: str, agent):
    """Testa un agente su un livello."""
    print(f"\n🧪 Test {agent_name}")
    
    start_time = time.time()
    try:
        # Parsa e crea il livello
        game_map = parse_map(level_data)
        initial_state = make_level(game_map)
        
        # Esegui l'agente
        actions = agent.search(initial_state, iterations=50000)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if actions:
            print(f"✅ Successo! Tempo: {execution_time:.3f}s, Mosse: {len(actions)}")
            return {'success': True, 'time': execution_time, 'moves': len(actions)}
        else:
            print(f"❌ Fallito. Tempo: {execution_time:.3f}s")
            return {'success': False, 'time': execution_time, 'moves': 0}
            
    except Exception as e:
        end_time = time.time()
        print(f"💥 Errore: {str(e)}")
        return {'success': False, 'time': end_time - start_time, 'moves': 0}

def main():
    """Test su livelli di diversa difficoltà."""
    print("🚀 Test comparativo agenti A* - Versioni Multiple")
    print("=" * 60)
    
    try:
        with open('json_levels/test_LEVELS.json', 'r') as f:
            data = json.load(f)
        
        # Test su livelli di diversa difficoltà
        test_indices = [0, 5, 10, 15, -5]  # Vari livelli di difficoltà
        
        for idx in test_indices:
            level = data['levels'][idx]
            level_ascii = level['ascii']
            level_id = level['id']
            expected_solution = level.get('solution', 'N/A')
            
            print(f"\n{'='*60}")
            print(f"📋 LIVELLO ID: {level_id}")
            print(f"🎯 Soluzione attesa: {expected_solution} ({len(expected_solution)} mosse)")
            print(f"{'='*60}")
            
            # Inizializza agenti
            agents = [
                ("A* Original", A_STARAgent()),
                ("A* Speed Optimized", OptimizedSpeedAgent()),
                ("A* V2 Optimized", OPTIMIZED_ASTAR_V2Agent())
            ]
            
            results = []
            
            # Test tutti gli agenti
            for agent_name, agent in agents:
                result = test_agent(level_ascii, agent_name, agent)
                result['agent'] = agent_name
                results.append(result)
                
                # Pulisci cache se disponibile
                if hasattr(agent, 'clear_caches'):
                    agent.clear_caches()
            
            # Confronto risultati
            print(f"\n📊 CONFRONTO LIVELLO {level_id}:")
            print("-" * 40)
            
            successful_results = [r for r in results if r['success']]
            
            if successful_results:
                # Ordina per tempo
                successful_results.sort(key=lambda x: x['time'])
                
                fastest = successful_results[0]
                print(f"🥇 Più veloce: {fastest['agent']} - {fastest['time']:.3f}s ({fastest['moves']} mosse)")
                
                for i, result in enumerate(successful_results[1:], 1):
                    slowdown = (result['time'] / fastest['time'] - 1) * 100
                    move_diff = result['moves'] - fastest['moves']
                    print(f"{'🥈' if i == 1 else '🥉'} {result['agent']}: {result['time']:.3f}s (+{slowdown:.1f}%) ({result['moves']} mosse, {move_diff:+d})")
            else:
                print("❌ Nessun agente è riuscito a completare il livello")
                for result in results:
                    print(f"   {result['agent']}: Fallito in {result['time']:.3f}s")
        
        # Statistiche finali
        print(f"\n{'='*60}")
        print("📈 STATISTICHE FINALI")
        print(f"{'='*60}")
        print("🏆 Test completato! Controlla i risultati sopra per le performance comparative.")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()
