/**
 * script.js - Script principale dell'interfaccia utente di Keke AI Python
 * 
 * Gestisce l'interfaccia utente, le richieste al server e utilizza il modulo
 * GameVisualization per mostrare le soluzioni dei livelli.
 */

document.addEventListener('DOMContentLoaded', () => {
    const agentSelect = document.getElementById('agent-select');
    const levelsetSelect = document.getElementById('levelset-select');
    const iterationsInput = document.getElementById('iterations-input');
    const solveBtn = document.getElementById('solve-btn');
    const statusMessage = document.getElementById('status-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    const levelsBody = document.getElementById('levels-body');
    const useCacheCheckbox = document.getElementById('use-cache');
    const clearCacheBtn = document.getElementById('clear-cache-btn');

    // Metriche
    const totalLevelsEl = document.getElementById('total-levels');
    const solvedLevelsEl = document.getElementById('solved-levels');
    const accuracyEl = document.getElementById('accuracy');
    const averageTimeEl = document.getElementById('average-time');
    const averageIterationsEl = document.getElementById('average-iterations');
    const averageEfficiencyEl = document.getElementById('average-efficiency');

    // Visualizzazione Gioco
    const gameViewContainer = document.getElementById('game-view-container');
    const gameLevelIdEl = document.getElementById('game-level-id');
    const gameCanvas = document.getElementById('game-canvas');
    const prevStepBtn = document.getElementById('prev-step-btn');
    const nextStepBtn = document.getElementById('next-step-btn');
    const playSolutionBtn = document.getElementById('play-solution-btn');
    const closeGameViewBtn = document.getElementById('close-game-view-btn');

    // Riferimenti agli elementi della barra di progresso
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');

    // Inizializza il modulo di visualizzazione del gioco
    GameVisualization.init({ canvas: gameCanvas });
    
    // Callback per la visualizzazione del gioco
    const visualizationCallbacks = {
        onStepUpdate: (currentStep, totalSteps) => {
            prevStepBtn.disabled = currentStep === 0;
            nextStepBtn.disabled = currentStep === totalSteps;
            playSolutionBtn.textContent = GameVisualization.getState().isPlaying ? "Pausa Soluzione" : "Avvia Soluzione";
        },
        onSolutionReady: (totalSteps) => {
            playSolutionBtn.disabled = false;
            prevStepBtn.disabled = true;
            nextStepBtn.disabled = totalSteps <= 1;
        },
        onNoSolution: () => {
            prevStepBtn.disabled = true;
            nextStepBtn.disabled = true;
            playSolutionBtn.disabled = true;
            console.warn("Nessuna soluzione disponibile per questo livello");
        },
        onPlaybackEnd: () => {
            playSolutionBtn.textContent = "Avvia Soluzione";
        }
    };

    function setStatus(message, isLoading) {
        statusMessage.textContent = message;
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        solveBtn.disabled = isLoading;
        
        // Nascondi la barra di progresso quando non c'è caricamento
        if (!isLoading) {
            progressContainer.style.display = 'none';
        }
    }
    
    function updateProgressBar(current, total) {
        // Mostra la barra di progresso
        progressContainer.style.display = 'block';
        
        // Aggiorna il testo della barra di progresso
        progressText.textContent = `${current}/${total}`;
        
        // Calcola la percentuale di completamento
        const percentage = (current / total) * 100;
        
        // Aggiorna la larghezza della barra di progresso
        progressBar.style.width = `${percentage}%`;
        
        // Cambia il colore della barra in base alla percentuale
        if (percentage < 30) {
            progressBar.style.backgroundColor = '#FF9800'; // Arancione
        } else if (percentage < 70) {
            progressBar.style.backgroundColor = '#2196F3'; // Blu
        } else {
            progressBar.style.backgroundColor = '#4CAF50'; // Verde
        }
    }

    async function loadInitialData() {
        setStatus("Caricamento dati iniziali...", true);
        try {
            const [agentsRes, levelSetsRes] = await Promise.all([
                fetch('/get_agents'),
                fetch('/get_level_sets')
            ]);
            const agents = await agentsRes.json();
            const levelSets = await levelSetsRes.json();

            agents.forEach(agent => {
                const option = document.createElement('option');
                option.value = agent.id;
                option.textContent = agent.name;
                agentSelect.appendChild(option);
            });

            levelSets.forEach(ls => {
                const option = document.createElement('option');
                option.value = ls.id;
                option.textContent = ls.name;
                levelsetSelect.appendChild(option);
            });
            setStatus("Pronto.", false);
        } catch (error) {
            console.error("Errore nel caricamento dati iniziali:", error);
            setStatus("Errore caricamento dati. Riprova.", false);
        }
    }

    function updateMetrics(report) {
        const summary = report.summary || {};
        const levels = report.levels || [];
        
        // Usa i valori del summary se disponibili, altrimenti calcolali
        if (summary.total_levels !== undefined && summary.solved_levels !== undefined) {
            console.log("Utilizzo metriche dal summary del backend:", summary);
            
            // Usa direttamente le metriche dal backend
            totalLevelsEl.textContent = summary.total_levels;
            solvedLevelsEl.textContent = summary.solved_levels;
            accuracyEl.textContent = summary.accuracy !== undefined ? 
                `${(summary.accuracy * 100).toFixed(1)}%` : '0%';
            averageTimeEl.textContent = summary.average_time !== undefined ? 
                `${summary.average_time.toFixed(3)}s` : '0s';
            averageIterationsEl.textContent = summary.average_iterations !== undefined ? 
                Math.round(summary.average_iterations) : '0';
            averageEfficiencyEl.textContent = summary.average_efficiency !== undefined ? 
                summary.average_efficiency.toFixed(4) : '0';
        } else {
            console.log("Calcolo metriche dai livelli:", levels);
            
            // Calcolo manuale se summary non completo
            let solvedCount = 0;
            let totalTime = 0;
            let totalIterations = 0;
            let totalEfficiency = 0;

            levels.forEach(level => {
                if (level.won_level) {
                    solvedCount++;
                    totalTime += level.time || 0;
                    totalIterations += level.iterations || 0;
                    totalEfficiency += level.efficiency_score || 0;
                }
            });

            totalLevelsEl.textContent = levels.length;
            solvedLevelsEl.textContent = solvedCount;
            accuracyEl.textContent = levels.length > 0 ? `${((solvedCount / levels.length) * 100).toFixed(1)}%` : '0%';
            averageTimeEl.textContent = solvedCount > 0 ? `${(totalTime / solvedCount).toFixed(3)}s` : '0s';
            averageIterationsEl.textContent = solvedCount > 0 ? Math.round(totalIterations / solvedCount) : '0';
            averageEfficiencyEl.textContent = solvedCount > 0 ? (totalEfficiency / solvedCount).toFixed(4) : '0';
        }
    }

    function displayLevels(levels) {
        levelsBody.innerHTML = ''; // Pulisce la tabella
        if (!levels || levels.length === 0) {
            const row = levelsBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 7; // Numero di colonne nella tabella (ID, Stato, Iterazioni, Tempo, Soluzione, Efficienza, Visualizza)
            cell.textContent = "Nessun livello da mostrare. Seleziona un agente e un set di livelli, poi clicca 'Risolvi Livelli'.";
            cell.style.textAlign = 'center';
            return;
        }
        
        // Log per debug
        console.log(`Visualizzazione di ${levels.length} livelli nella tabella`);

        levels.forEach(level => {
            const row = levelsBody.insertRow();
            row.insertCell().textContent = level.id;
            
            const statusCell = row.insertCell();
            statusCell.textContent = level.won_level ? 'Risolto' : (level.status === 'pending' ? 'In attesa' : 'Non Risolto');
            statusCell.className = level.won_level ? 'status-solved' : (level.status === 'pending' ? 'status-pending' : 'status-unsolved');
            
            row.insertCell().textContent = level.iterations || 0;
            row.insertCell().textContent = (level.time || 0).toFixed(3);
            row.insertCell().textContent = level.solution || '-';
            
            // Aggiungi efficienza con formattazione in base al valore
            const efficiencyCell = row.insertCell();
            const efficiencyValue = level.efficiency_score;
            
            if (efficiencyValue && level.won_level) {
                // Aggiungi classe per lo stile in base all'efficienza
                let efficiencyClass = 'efficiency-low';
                if (efficiencyValue >= 0.5) efficiencyClass = 'efficiency-excellent';
                else if (efficiencyValue >= 0.2) efficiencyClass = 'efficiency-good';
                else if (efficiencyValue >= 0.05) efficiencyClass = 'efficiency-average';
                
                efficiencyCell.textContent = efficiencyValue.toFixed(4);
                efficiencyCell.className = efficiencyClass;
            } else {
                efficiencyCell.textContent = '-';
            }

            const actionCell = row.insertCell();
            const viewBtn = document.createElement('button');
            viewBtn.textContent = "Visualizza";
            viewBtn.className = 'view-level-btn';
            viewBtn.onclick = async () => {
                // Qui recuperiamo i dettagli completi del livello, inclusa la mappa ASCII aggiornata se necessario
                if (level.ascii_map) {
                    showGameView(level.id, level.ascii_map, level.solution);
                } else {
                    // Se ascii_map non è nel report principale, la carichiamo
                    setStatus(`Caricamento mappa per ${level.id}...`, true);
                    try {
                        const response = await fetch(`/get_level_details/${levelsetSelect.value}/${level.id}`);
                        if (!response.ok) throw new Error(`Errore HTTP: ${response.status}`);
                        const details = await response.json();
                        if (details.error) throw new Error(details.error);
                        showGameView(details.id, details.ascii_map, level.solution); // Usa la soluzione dal report generale
                    } catch (err) {
                        console.error("Errore caricamento dettagli livello:", err);
                        alert("Impossibile caricare i dettagli del livello: " + err.message);
                    } finally {
                        setStatus("Pronto.", false);
                    }
                }
            };
            actionCell.appendChild(viewBtn);
        });
    }

    // Funzione per mostrare la visualizzazione del gioco
    async function showGameView(levelId, asciiMap, solution) {
        // Mostra il container della visualizzazione
        gameViewContainer.style.display = 'flex';
        gameLevelIdEl.textContent = levelId;
        
        // Usa il modulo GameVisualization per visualizzare il gioco
        await GameVisualization.showGameView(levelId, asciiMap, solution, gameCanvas, visualizationCallbacks);
    }

    let progressEventSource = null;
    
    solveBtn.addEventListener('click', async () => {
        const agentId = agentSelect.value;
        const levelSetId = levelsetSelect.value;
        const iterations = parseInt(iterationsInput.value);
        const useCache = useCacheCheckbox.checked;

        if (!agentId || !levelSetId) {
            alert("Seleziona un agente e un set di livelli.");
            return;
        }
        
        // Chiude eventuali EventSource esistenti
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }

        setStatus(`Avvio risoluzione con ${agentId} su ${levelSetId}${useCache ? ' (con cache)' : ''}...`, true);
        levelsBody.innerHTML = ''; // Pulisce la tabella prima di iniziare
        totalLevelsEl.textContent = '-';
        solvedLevelsEl.textContent = '-';
        accuracyEl.textContent = '-';
        averageTimeEl.textContent = '-';
        averageIterationsEl.textContent = '-';
        averageEfficiencyEl.textContent = '-';
        gameViewContainer.style.display = 'none'; // Nasconde la vista gioco
        progressContainer.style.display = 'block'; // Mostra il container della barra di progresso

        try {
            // Inizia l'elaborazione sul server e ottieni l'ID di sessione
            const initResponse = await fetch('/solve_level_set', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ agent: agentId, levelSet: levelSetId, iterations, useCache })
            });

            if (!initResponse.ok) {
                const errorData = await initResponse.json().catch(() => null);
                throw new Error(`Errore HTTP: ${initResponse.status}${errorData ? ' - ' + (errorData.error || JSON.stringify(errorData)) : ''}`);
            }

            const initData = await initResponse.json();
            if (initData.error) {
                throw new Error(initData.error);
            }
            
            const sessionId = initData.session_id;
            console.log(`Sessione avviata con ID: ${sessionId}`);
            
            // Configura l'EventSource per ricevere aggiornamenti in tempo reale
            progressEventSource = new EventSource(`/progress/${sessionId}`);
            
            // Gestisci gli eventi di progresso
            progressEventSource.onmessage = (event) => {
                const progressData = JSON.parse(event.data);
                
                // Se lo stato è "completed", significa che l'elaborazione è terminata
                if (progressData.status === 'completed') {
                    // Chiudi la connessione SSE
                    progressEventSource.close();
                    progressEventSource = null;
                    
                    // Aggiorna lo stato
                    setStatus("Risoluzione completata.", false);
                    return;
                }
                
                // Se lo stato è "error", gestisci l'errore
                if (progressData.status === 'error') {
                    progressEventSource.close();
                    progressEventSource = null;
                    
                    console.error("Errore durante l'elaborazione:", progressData.error);
                    setStatus(`Errore: ${progressData.error}`, false);
                    alert(`Si è verificato un errore: ${progressData.error}`);
                    return;
                }
                
                // Gestisci gli aggiornamenti di progresso normali
                const current = progressData.current;
                const total = progressData.total;
                const report = progressData.report;
                const fromCache = report && report.from_cache;
                
                // Log per debug
                console.log(`Progresso: ${current}/${total} (${((current/total)*100).toFixed(1)}%)`);
                
                // Aggiorna la barra di progresso
                updateProgressBar(current, total);
                
                // Aggiorna lo stato
                if (fromCache) {
                    setStatus(`Caricamento risultato dalla cache: ${current}/${total}...`, true);
                } else {
                    setStatus(`Elaborazione livello ${current}/${total}...`, true);
                }
                
                // Aggiorna la visualizzazione dei livelli e le metriche
                if (report && report.levels) {
                    displayLevels(report.levels);
                    updateMetrics(report);
                    
                    // Aggiungi un indicatore se i risultati provengono dalla cache
                    if (fromCache) {
                        statusMessage.innerHTML = `Risultati caricati dalla cache: ${current}/${total} <span class="from-cache">(dalla cache)</span>`;
                    }
                }
            };
            
            // Gestisci gli errori dell'EventSource
            progressEventSource.onerror = (error) => {
                console.error("Errore nella connessione SSE:", error);
                
                // Chiudi la connessione e mostra un messaggio di errore
                progressEventSource.close();
                progressEventSource = null;
                setStatus("Errore nella connessione col server. Riprova.", false);
            };
            
        } catch (error) {
            console.error("Errore durante l'avvio della risoluzione:", error);
            setStatus(`Errore: ${error.message}`, false);
            alert(`Si è verificato un errore: ${error.message}`);
        }
    });

    // Gestione degli eventi per i pulsanti di controllo del gioco
    closeGameViewBtn.addEventListener('click', () => {
        gameViewContainer.style.display = 'none';
        GameVisualization.cleanup();
    });

    prevStepBtn.addEventListener('click', () => {
        const state = GameVisualization.getState();
        if (state.currentStep > 0) {
            GameVisualization.displayStep(state.currentStep - 1, visualizationCallbacks, gameCanvas);
        }
    });

    nextStepBtn.addEventListener('click', () => {
        const state = GameVisualization.getState();
        if (state.hasSolution && state.currentStep < state.totalSteps) {
            GameVisualization.displayStep(state.currentStep + 1, visualizationCallbacks, gameCanvas);
        }
    });

    playSolutionBtn.addEventListener('click', () => {
        const isPlaying = GameVisualization.toggleSolutionPlayback(
            visualizationCallbacks,
            gameCanvas,
            300 // velocità in millisecondi
        );
        playSolutionBtn.textContent = isPlaying ? "Pausa Soluzione" : "Avvia Soluzione";
    });

    // Funzione per pulire la cache
    function clearCache() {
        statusMessage.textContent = "Pulizia della cache in corso...";
        loadingIndicator.style.display = "block";
        
        fetch('/manage_cache', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ action: 'clear_all' }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                statusMessage.textContent = data.message;
            } else {
                statusMessage.textContent = `Errore: ${data.error || 'Impossibile pulire la cache'}`;
            }
            loadingIndicator.style.display = "none";
        })
        .catch(error => {
            console.error('Errore durante la pulizia della cache:', error);
            statusMessage.textContent = "Errore durante la pulizia della cache";
            loadingIndicator.style.display = "none";
        });
    }
    
    // Registra l'evento di click per il pulsante di pulizia della cache
    clearCacheBtn.addEventListener('click', clearCache);

    // Registra l'evento di click per il pulsante della classifica
    const leaderboardBtn = document.getElementById('leaderboard-btn');
    leaderboardBtn.addEventListener('click', () => {
        console.log('🏆 Apertura classifica agenti...');
        Leaderboard.show();
    });

    // Inizializzazione
    loadInitialData();
    GameVisualization.preloadSprites().then(() => {
        console.log("Sprite caricati, pronto per disegnare mappe.");
        // Pulisce il canvas o mostra un messaggio di default
        GameVisualization.drawEmptyMap(gameCanvas);
    });
    displayLevels([]); // Mostra messaggio iniziale nella tabella
});
