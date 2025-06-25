/**
 * Modulo per la gestione della Classifica degli Agenti
 */
const Leaderboard = (function() {
    let leaderboardData = [];
    let filteredData = [];
    
    // Elementi DOM
    let modal = null;
    let tableBody = null;
    let levelSetFilter = null;
    let sortByFilter = null;
    let statsContainer = null;

    /**
     * Inizializza il modulo della classifica
     */
    function init() {
        createModalHTML();
        attachEventListeners();
        console.log('📊 Modulo Classifica inizializzato');
    }

    /**
     * Crea la struttura HTML del modal
     */
    function createModalHTML() {
        const modalHTML = `
            <div id="leaderboard-modal" class="leaderboard-modal">
                <div class="leaderboard-modal-content">
                    <div class="leaderboard-header">
                        <h2>🏆 Classifica Agenti</h2>
                        <button class="leaderboard-close" onclick="Leaderboard.close()">&times;</button>
                    </div>
                    <div class="leaderboard-body">
                        <div id="leaderboard-stats" class="leaderboard-stats">
                            <!-- Le statistiche verranno inserite qui -->
                        </div>
                        
                        <div class="leaderboard-filters">
                            <div class="leaderboard-filter">
                                <label for="level-set-filter">Filtra per Set di Livelli:</label>
                                <select id="level-set-filter">
                                    <option value="all">Tutti i Set</option>
                                </select>
                            </div>
                            <div class="leaderboard-filter">
                                <label for="sort-by-filter">Ordina per:</label>
                                <select id="sort-by-filter">
                                    <option value="accuracy">Accuratezza</option>
                                    <option value="time">Tempo Medio</option>
                                    <option value="iterations">Iterazioni Medie</option>
                                    <option value="solved">Livelli Risolti</option>
                                    <option value="efficiency">Efficienza (Tempo^-1 / Lunghezza)</option>
                                </select>
                            </div>
                        </div>

                        <div class="leaderboard-table-container">
                            <table class="leaderboard-table">
                                <thead>
                                    <tr>
                                        <th>Pos.</th>
                                        <th>Agente</th>
                                        <th>Set Livelli</th>
                                        <th>Accuratezza</th>
                                        <th>Risolti / Totali</th>
                                        <th>Tempo Medio</th>
                                        <th>Iter. Medie</th>
                                        <th>Efficienza</th>
                                        <th>Esecuzioni</th>
                                    </tr>
                                </thead>
                                <tbody id="leaderboard-table-body">
                                    <!-- Le righe verranno inserite qui -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Ottieni riferimenti agli elementi
        modal = document.getElementById('leaderboard-modal');
        tableBody = document.getElementById('leaderboard-table-body');
        levelSetFilter = document.getElementById('level-set-filter');
        sortByFilter = document.getElementById('sort-by-filter');
        statsContainer = document.getElementById('leaderboard-stats');
    }

    /**
     * Aggiunge i listener agli eventi
     */
    function attachEventListeners() {
        // Chiudi il modal cliccando fuori
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                close();
            }
        });

        // Filtri
        levelSetFilter.addEventListener('change', applyFilters);
        sortByFilter.addEventListener('change', applyFilters);

        // Chiudi con ESC
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                close();
            }
        });
    }

    /**
     * Mostra la classifica
     */
    async function show() {
        console.log('📊 Caricamento classifica...');
        
        try {
            const response = await fetch('/get_leaderboard');
            const data = await response.json();
            
            if (data.status === 'success') {
                leaderboardData = data.leaderboard || [];
                updateFilters();
                applyFilters();
                updateStats();
                modal.style.display = 'block';
                console.log(`📊 Classifica caricata: ${leaderboardData.length} agenti`);
            } else {
                console.error('❌ Errore nel caricamento classifica:', data.error);
                alert('Errore nel caricamento della classifica: ' + data.error);
            }
        } catch (error) {
            console.error('❌ Errore nella richiesta classifica:', error);
            alert('Errore nella connessione al server');
        }
    }

    /**
     * Chiude la classifica
     */
    function close() {
        modal.style.display = 'none';
        console.log('📊 Classifica chiusa');
    }

    /**
     * Aggiorna i filtri disponibili
     */
    function updateFilters() {
        // Ottieni tutti i set di livelli unici
        const levelSets = [...new Set(leaderboardData.map(item => item.level_set))];
        
        // Aggiorna il filtro dei set di livelli
        levelSetFilter.innerHTML = '<option value="all">Tutti i Set</option>';
        levelSets.forEach(levelSet => {
            const option = document.createElement('option');
            option.value = levelSet;
            option.textContent = levelSet;
            levelSetFilter.appendChild(option);
        });
    }

    /**
     * Applica i filtri e l'ordinamento
     */
    function applyFilters() {
        const selectedLevelSet = levelSetFilter.value;
        const sortBy = sortByFilter.value;
        
        // Filtra i dati
        filteredData = leaderboardData.filter(item => {
            if (selectedLevelSet === 'all') return true;
            return item.level_set === selectedLevelSet;
        });
        
        // Ordina i dati
        filteredData.sort((a, b) => {
            switch (sortBy) {
                case 'accuracy':
                    return b.accuracy - a.accuracy || a.avg_time - b.avg_time;
                case 'time':
                    return a.avg_time - b.avg_time;
                case 'iterations':
                    return a.avg_iterations - b.avg_iterations;
                case 'solved':
                    return b.solved_levels - a.solved_levels || b.accuracy - a.accuracy;
                case 'efficiency':
                    // I valori più alti di efficienza sono migliori
                    // Per gestire i valori null/undefined, controlliamo esplicitamente
                    const aEffNull = a.avg_efficiency === null || a.avg_efficiency === undefined;
                    const bEffNull = b.avg_efficiency === null || b.avg_efficiency === undefined;
                    
                    if (aEffNull && bEffNull) return 0;
                    if (aEffNull) return 1; // b è migliore
                    if (bEffNull) return -1; // a è migliore
                    return b.avg_efficiency - a.avg_efficiency;
                default:
                    return b.accuracy - a.accuracy;
            }
        });
        
        renderTable();
    }

    /**
     * Renderizza la tabella della classifica
     */
    function renderTable() {
        if (filteredData.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="no-data-message">
                        <i>📭</i>
                        Nessun dato disponibile per i filtri selezionati
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = '';
        
        filteredData.forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Determina la classe del rank
            let rankClass = '';
            if (index === 0) rankClass = 'rank-1';
            else if (index === 1) rankClass = 'rank-2';
            else if (index === 2) rankClass = 'rank-3';
            
            // Determina la classe dell'accuratezza
            let accuracyClass = 'accuracy-poor';
            if (item.accuracy >= 90) accuracyClass = 'accuracy-excellent';
            else if (item.accuracy >= 70) accuracyClass = 'accuracy-good';
            else if (item.accuracy >= 50) accuracyClass = 'accuracy-average';
            
            // Aggiungi una classe per l'efficienza
            let efficiencyClass = 'efficiency-low';
            if (item.avg_efficiency !== null && item.avg_efficiency !== undefined) {
                if (item.avg_efficiency >= 0.5) efficiencyClass = 'efficiency-excellent';
                else if (item.avg_efficiency >= 0.2) efficiencyClass = 'efficiency-good';
                else if (item.avg_efficiency >= 0.05) efficiencyClass = 'efficiency-average';
            }
            
            row.innerHTML = `
                <td class="rank-cell ${rankClass}">${index + 1}</td>
                <td class="agent-name">${item.agent}</td>
                <td class="level-set-name">${item.level_set}</td>
                <td class="accuracy-cell ${accuracyClass}">${item.accuracy.toFixed(1)}%</td>
                <td class="solved-cell">${item.solved_levels} / ${item.total_levels}</td>
                <td class="time-cell">${item.avg_time.toFixed(3)}s</td>
                <td class="iterations-cell">${Math.round(item.avg_iterations)}</td>
                <td class="efficiency-cell ${efficiencyClass}">${item.avg_efficiency !== undefined && item.avg_efficiency !== null ? item.avg_efficiency.toFixed(4) : "N/A"}</td>
                <td><span class="executions-badge">${item.total_executions}</span></td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    /**
     * Aggiorna le statistiche generali
     */
    function updateStats() {
        const totalAgents = leaderboardData.length;
        const totalExecutions = leaderboardData.reduce((sum, item) => sum + item.total_executions, 0);
        const avgAccuracy = leaderboardData.length > 0 
            ? leaderboardData.reduce((sum, item) => sum + item.accuracy, 0) / leaderboardData.length 
            : 0;
            
        // Calcolo dell'efficienza media globale
        let validEfficiencyCount = 0;
        const avgEfficiency = leaderboardData.length > 0 
            ? leaderboardData.reduce((sum, item) => {
                if (item.avg_efficiency !== undefined && item.avg_efficiency !== null) {
                    validEfficiencyCount++;
                    return sum + item.avg_efficiency;
                }
                return sum;
              }, 0) / (validEfficiencyCount || 1) 
            : 0;
            
        const bestAgent = leaderboardData.length > 0 ? leaderboardData[0] : null;
        
        statsContainer.innerHTML = `
            <div class="leaderboard-stat">
                <span class="leaderboard-stat-value">${totalAgents}</span>
                <span class="leaderboard-stat-label">Agenti Testati</span>
            </div>
            <div class="leaderboard-stat">
                <span class="leaderboard-stat-value">${totalExecutions}</span>
                <span class="leaderboard-stat-label">Esecuzioni Totali</span>
            </div>
            <div class="leaderboard-stat">
                <span class="leaderboard-stat-value">${avgAccuracy.toFixed(1)}%</span>
                <span class="leaderboard-stat-label">Accuratezza Media</span>
            </div>
            <div class="leaderboard-stat">
                <span class="leaderboard-stat-value">${validEfficiencyCount > 0 ? avgEfficiency.toFixed(4) : 'N/A'}</span>
                <span class="leaderboard-stat-label">Efficienza Media</span>
            </div>
            <div class="leaderboard-stat">
                <span class="leaderboard-stat-value">${bestAgent ? bestAgent.agent : 'N/A'}</span>
                <span class="leaderboard-stat-label">Miglior Agente</span>
            </div>
        `;
    }

    // API pubblica
    return {
        init,
        show,
        close
    };
})();

// Inizializza il modulo quando il DOM è pronto
document.addEventListener('DOMContentLoaded', () => {
    Leaderboard.init();
});
