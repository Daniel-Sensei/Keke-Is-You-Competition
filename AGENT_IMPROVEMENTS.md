# Miglioramenti degli Agenti per Baba Is You - KekeAI

## Panoramica delle Migliorie

Basandoci sulle regole specifiche di Baba Is You e sull'analisi del codice esistente, sono stati sviluppati tre nuovi agenti con euristiche avanzate e strategie migliorate.

## 1. Improved A* Agent (`improved_astar_AGENT.py`)

### Principali Migliorie:

#### Sistema di Priorità delle Regole
- **YOU rules**: Massima priorità (peso 5.0) - controllo del giocatore
- **WIN rules**: Seconda priorità (peso 4.0) - condizioni di vittoria  
- **Transformation rules**: Terza prioritità (peso 3.0) - regole X IS Y
- **Property rules**: Quarta priorità (peso 2.0) - STOP, PUSH, HOT, etc.
- **Blocking rules**: Quinta priorità (peso 1.5) - regole X IS X

#### Analisi delle Dipendenze delle Regole
```python
def _analyze_rule_dependencies(self, state: GameState) -> Dict[str, float]:
```
- Identifica automaticamente quali regole sono più critiche per vincere
- Considera gli oggetti presenti sulla mappa
- Assegna punteggi di importanza dinamici

#### Rilevazione dei Conflitti tra Regole
- Detecta regole contradditorie (es. "BABA IS ROCK" e "BABA IS FLAG")
- Identifica regole riflessive (X IS X) che bloccano trasformazioni necessarie
- Applica penalità per stati con conflitti

#### Analisi dei Percorsi di Trasformazione
- Stima il costo delle catene di trasformazione necessarie
- Considera la vicinanza tra giocatore e oggetti vincenti
- Evalua se trasformazioni intermedie possono migliorare il percorso

#### Euristica Potenziata Multi-Componente
```python
def _calculate_enhanced_heuristic(self, state: GameState) -> float:
```
Combina:
1. **Distanza base** (peso 1.0)
2. **Formazione regole** (peso 2.5) 
3. **Interazioni regole** (peso 1.8)
4. **Fattore complessità dinamico**

## 2. Improved Evolutionary Agent (`improved_evolutionary_AGENT.py`)

### Principali Migliorie:

#### Fitness Multi-Obiettivo Avanzato
- **Componente obiettivo** (peso 0.3): progresso verso la vittoria
- **Componente novità** (peso 0.4): esplorazione di nuovi comportamenti
- **Progresso regole** (peso 0.3): progresso nella formazione di regole

#### Caratterizzazione Comportamentale Potenziata
```python
def _get_enhanced_behavior_characterization(self, state: GameState, sequence: List[Direction]) -> Dict:
```
Include:
- Posizione del giocatore
- Insieme di regole attive
- Punteggio di allineamento delle parole
- Diversità della sequenza di azioni
- Pattern di esplorazione

#### Mutazione Adattiva
```python
def _adaptive_mutation(self, sequence: List[Direction], generation: int) -> List[Direction]:
```
- Tasso di mutazione che si adatta in base al progresso
- Aumenta mutazione durante stagnazione
- Diminuisce mutazione quando si fanno progressi

#### Analisi del Progresso nella Formazione delle Regole
- Traccia la storia delle formazioni di regole per novità
- Bonus per allineamento delle parole che potrebbero formare regole
- Punteggi per diversità delle regole (evita ottimi locali)

#### Conservazione Elite con Diversità
- Preserva i migliori individui
- Mantiene diversità comportamentale nell'archivio
- Selezione a torneo migliorata

## 3. Hybrid Agent (`hybrid_AGENT.py`)

### Approccio Innovativo:

#### Commutazione Dinamica degli Algoritmi
```python
def _analyze_problem_characteristics(self, initial_state: GameState) -> Dict[str, Any]:
```
Analizza:
- **Complessità del problema**: numero di oggetti, parole, regole
- **Necessità di formazione regole**: mancanza di YOU/WIN rules
- **Intensività esplorativa**: mappe grandi, molti oggetti

#### Strategia di Ricerca a Fasi
1. **Fase 1**: Metodo preferito basato sull'analisi
2. **Fase 2**: Metodo complementare
3. **Fase 3**: Ricerca A* potenziata con conoscenza evolutiva

#### Condivisione della Conoscenza
```python
self.shared_knowledge = {
    'promising_rule_patterns': set(),
    'effective_action_sequences': [],
    'dead_end_states': set(),
    'successful_strategies': []
}
```

#### Popolazione Seeded per l'Evolutivo
- Usa sequenze efficaci trovate da A* come building blocks
- Migliora convergenza dell'algoritmo evolutivo
- Combina esplorazione e sfruttamento

## Miglioramenti Specifici per Baba Is You

### 1. Comprensione delle Regole del Gioco
- **X IS KEYWORD**: Gestione corretta delle proprietà (MOVE, STOP, HOT, etc.)
- **X IS Y**: Trasformazioni con analisi dell'impatto
- **X IS X**: Regole riflessive che prevengono trasformazioni

### 2. Prioritizzazione Intelligente
```
YOU > WIN > Trasformazioni > Proprietà > Blocco
```

### 3. Analisi delle Catene di Trasformazione
- Identifica percorsi multi-step per raggiungere condizioni di vittoria
- Evita cicli infiniti di trasformazione
- Ottimizza l'ordine delle trasformazioni

### 4. Gestione della Complessità
- Adattamento dinamico dei parametri basato sulla complessità del livello
- Bilanciamento tra esplorazione e sfruttamento
- Gestione efficiente della memoria e cache

## Parametri Ottimizzati

### A* Migliorato:
- Iterazioni aumentate a 100-150
- Pesi euristici calibrati per Baba Is You
- Penalità dinamiche basate su complessità

### Evolutivo Migliorato:
- Popolazione: 50 individui
- Generazioni: 150
- Lunghezza soluzione: 120 mosse
- Ratio elite: 15%

### Ibrido:
- Tempo massimo per metodo: 30 secondi
- Soglia di commutazione: 50 iterazioni
- Popolazione evolutiva ridotta: 30 individui per velocità

## Vantaggi Rispetto agli Agenti Originali

1. **Comprensione del dominio**: Euristiche specifiche per Baba Is You
2. **Adattabilità**: Parametri che si adattano al problema
3. **Robustezza**: Gestione di casi edge e conflitti tra regole
4. **Efficienza**: Migliore bilanciamento esplorazione/sfruttamento
5. **Scalabilità**: Gestione di livelli complessi con molte regole

## Utilizzo

Gli agenti possono essere testati direttamente dall'interfaccia web:
- `improved_astar`: A* con euristiche avanzate
- `improved_evolutionary`: Evolutivo multi-obiettivo
- `hybrid`: Combinazione dinamica dei due approcci

Si consiglia di testare tutti e tre gli approcci su diversi tipi di livelli per valutare quale funziona meglio per specifiche classi di problemi.
