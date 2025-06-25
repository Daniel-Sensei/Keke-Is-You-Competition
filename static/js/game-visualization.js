/**
 * game-visualization.js
 * Gestisce la visualizzazione del gioco Baba Is You e l'animazione delle mosse
 */

// Modulo per la visualizzazione del gioco
const GameVisualization = (function() {
    // Costanti e variabili private
    const TILE_SIZE = 20; // Dimensione di una cella della griglia
    const SPRITE_IMAGES = {}; // Cache per le immagini degli sprite
    const SPRITE_BASE_PATH = '/static/img/'; // Percorso alle immagini degli sprite
    
    let currentSolution = [];
    let currentStepIndex = -1;
    let solutionPlayerInterval = null;
    let ctx = null;
    let precomputedStates = []; // Stati del gioco pre-calcolati dal server
    
    // Mappatura dei caratteri della mappa agli sprite
    const MAP_SPRITES = {
        // Oggetti di gioco
        '_': 'border.png',
        ' ': 'empty.png',
        '.': 'floor_obj.png',
        'b': 'baba_obj.png',
        'B': 'baba_word.png',
        's': 'skull_obj.png',
        'S': 'skull_word.png',
        'f': 'flag_obj.png',
        'F': 'flag_word.png',
        'o': 'floor_obj.png',
        'O': 'floor_word.png',
        'a': 'grass_obj.png',
        'A': 'grass_word.png',
        'l': 'lava_obj.png',
        'L': 'lava_word.png',
        'r': 'rock_obj.png',
        'R': 'rock_word.png',
        'w': 'wall_obj.png',
        'W': 'wall_word.png',
        'k': 'keke_obj.png',
        'K': 'keke_word.png',
        'g': 'goop_obj.png',
        'G': 'goop_word.png',
        'v': 'love_obj.png',
        'V': 'love_word.png',
        
        // Parole
        '1': 'is_word.png',
        '2': 'you_word.png',
        '3': 'win_word.png',
        '4': 'kill_word.png',
        '5': 'push_word.png',
        '6': 'stop_word.png',
        '7': 'move_word.png',
        '8': 'hot_word.png',
        '9': 'melt_word.png',
        '0': 'sink_word.png'
    };

    // Funzioni private
    async function loadSprite(spriteName) {
        if (SPRITE_IMAGES[spriteName]) {
            return SPRITE_IMAGES[spriteName];
        }
        if (!MAP_SPRITES[spriteName] && !spriteName.endsWith('.png')) {
            console.warn(`Sprite non mappato: ${spriteName}`);
            return null; // O un'immagine di default
        }
        const imgSrc = spriteName.endsWith('.png') ? SPRITE_BASE_PATH + spriteName : SPRITE_BASE_PATH + MAP_SPRITES[spriteName];
        
        try {
            const img = new Image();
            img.src = imgSrc;
            await img.decode(); // Attende che l'immagine sia caricata e decodificata
            SPRITE_IMAGES[spriteName] = img;
            return img;
        } catch (error) {
            console.error(`Errore caricamento sprite ${spriteName} da ${imgSrc}:`, error);
            return null; // O un'immagine di errore/default
        }
    }
    
    function parseAsciiMap(asciiMap) {
        if (!asciiMap) return [];
        return asciiMap.trim().split('\n').map(row => row.split(''));
    }
    
    async function drawMap(mapArray, canvas) {
        if (!ctx) {
            ctx = canvas.getContext('2d');
        }
        
        if (!mapArray || mapArray.length === 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#333';
            ctx.textAlign = 'center';
            ctx.fillText("Mappa non disponibile o vuota.", canvas.width / 2, canvas.height / 2);
            return;
        }

        const numRows = mapArray.length;
        const numCols = mapArray[0].length;

        canvas.width = numCols * TILE_SIZE;
        canvas.height = numRows * TILE_SIZE;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let r = 0; r < numRows; r++) {
            for (let c = 0; c < numCols; c++) {
                const char = mapArray[r][c];
                const spriteImg = await loadSprite(char);
                if (spriteImg) {
                    ctx.drawImage(spriteImg, c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                } else {
                    // Fallback se lo sprite non è trovato
                    ctx.fillStyle = '#555';
                    ctx.fillRect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                }
            }
        }
    }
    
    /**
     * Stampa la mappa in console in formato leggibile per debug
     * @param {Array} mapArray Array 2D della mappa
     * @param {string} stepInfo Informazioni sullo step corrente
     */
    function logMapToConsole(mapArray, stepInfo) {
        if (!mapArray || mapArray.length === 0) {
            console.log(`${stepInfo}: Mappa vuota`);
            return;
        }
        
        console.log(`\n=== ${stepInfo} ===`);
        console.log('Mappa:');
        
        // Stampa la mappa riga per riga
        mapArray.forEach((row, rowIndex) => {
            const rowStr = row.join(' ');
            console.log(`${rowIndex.toString().padStart(2, '0')}: ${rowStr}`);
        });
        
        // Mostra leggenda caratteri presenti
        const uniqueChars = new Set();
        mapArray.forEach(row => {
            row.forEach(char => uniqueChars.add(char));
        });
        
        console.log('\nCaratteri presenti nella mappa:');
        Array.from(uniqueChars).sort().forEach(char => {
            const spriteName = MAP_SPRITES[char] || 'NON_MAPPATO';
            console.log(`  '${char}' -> ${spriteName}`);
        });
        console.log('========================\n');
    }
    
    /**
     * Salva la risoluzione step-by-step in formato JSON
     * @param {Array} states Array degli stati pre-calcolati
     * @param {string} levelId ID del livello
     */
    async function saveStepByStepSolution(states, levelId) {
        if (!states || states.length === 0) {
            console.log("Nessuno stato da salvare");
            return;
        }
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `solution_${levelId}_${timestamp}.json`;
        
        const solutionData = {
            levelId: levelId,
            timestamp: new Date().toISOString(),
            totalSteps: states.length - 1, // -1 perché il primo è lo stato iniziale
            states: states.map((state, index) => ({
                step: index,
                move: state.move || (index === 0 ? 'INITIAL' : 'UNKNOWN'),
                ascii_map: state.ascii_map,
                description: index === 0 ? 'Stato iniziale' : `Dopo mossa: ${state.move}`
            }))
        };
        
        try {
            // Invia i dati al server per il salvataggio
            const response = await fetch('/save_solution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: filename,
                    data: solutionData
                })
            });
            
            if (response.ok) {
                console.log(`✅ Soluzione salvata: ${filename}`);
            } else {
                console.error(`❌ Errore nel salvataggio: ${response.statusText}`);
            }
        } catch (error) {
            console.error('❌ Errore nel salvataggio della soluzione:', error);
            
            // Fallback: salva in localStorage se il server non risponde
            try {
                const jsonString = JSON.stringify(solutionData, null, 2);
                localStorage.setItem(`solution_${levelId}_${timestamp}`, jsonString);
                console.log(`💾 Soluzione salvata in localStorage: solution_${levelId}_${timestamp}`);
            } catch (storageError) {
                console.error('❌ Errore anche nel salvataggio localStorage:', storageError);
            }
        }
    }

    /**
     * Visualizza uno step specifico usando gli stati pre-calcolati dal server
     * @param {number} stepIndex Indice dello step da visualizzare
     * @param {Object} callbacks Funzioni di callback  
     * @param {HTMLCanvasElement} canvas Canvas su cui disegnare
     */
    async function displayPrecomputedStep(stepIndex, callbacks, canvas) {
        if (!precomputedStates || precomputedStates.length === 0) {
            console.log('❌ DEBUG: Nessuno stato pre-calcolato disponibile');
            return;
        }
        if (stepIndex < 0 || stepIndex >= precomputedStates.length) {
            console.log(`❌ DEBUG: stepIndex ${stepIndex} fuori range [0, ${precomputedStates.length - 1}]`);
            return;
        }

        const state = precomputedStates[stepIndex];
        console.log(`🔍 DEBUG: Visualizzazione step ${stepIndex}/${precomputedStates.length - 1}: ${state.move || 'INITIAL'}`);
        
        currentStepIndex = stepIndex;
        
        // Usa lo stato pre-calcolato per disegnare la mappa
        const mapArray = parseAsciiMap(state.ascii_map);
        
        // 🔍 DEBUG: Stampa la mappa in console per verificare la mappatura
        const stepInfo = stepIndex === 0 
            ? 'STATO INIZIALE' 
            : `STEP ${stepIndex} - Mossa: ${state.move}`;
        logMapToConsole(mapArray, stepInfo);
        
        await drawMap(mapArray, canvas);
        
        // Aggiorna lo stato dei controlli attraverso i callback
        // Il numero di mosse è precomputedStates.length - 1 (escludendo lo stato iniziale)
        if (callbacks.onStepUpdate) {
            callbacks.onStepUpdate(stepIndex, precomputedStates.length - 1);
        }
    }

    // API pubblica
    return {
        /**
         * Inizializza il modulo di visualizzazione
         * @param {Object} config Configurazione iniziale
         */
        init: function(config) {
            if (config.canvas) {
                ctx = config.canvas.getContext('2d');
            }
            // Reset delle variabili
            currentSolution = [];
            currentStepIndex = -1;
            if (solutionPlayerInterval) {
                clearInterval(solutionPlayerInterval);
                solutionPlayerInterval = null;
            }
            
            return this;
        },
        
        /**
         * Precarica tutti gli sprite necessari
         */
        preloadSprites: async function() {
            const promises = [];
            for (const key in MAP_SPRITES) {
                promises.push(loadSprite(MAP_SPRITES[key]));
            }
            await Promise.all(promises);
            console.log("Sprite precaricati.");
            return this;
        },
        
        /**
         * Visualizza un livello di gioco
         * @param {string} levelId ID del livello
         * @param {string} asciiMap Mappa ASCII del livello
         * @param {Array|string} solution Soluzione da visualizzare
         * @param {HTMLCanvasElement} canvas Canvas su cui disegnare
         * @param {Object} callbacks Funzioni di callback per eventi
         */
        showGameView: async function(levelId, asciiMap, solution, canvas, callbacks = {}) {
            console.log("showGameView - solution:", solution, "type:", typeof solution);
            
            // Interrompi qualsiasi animazione esistente
            if (solutionPlayerInterval) {
                clearInterval(solutionPlayerInterval);
                solutionPlayerInterval = null;
            }
            
            // Reset delle variabili locali
            currentStepIndex = -1;
            precomputedStates = [];
            
            // Converti la soluzione in array se necessario
            let solutionArray = [];
            if (Array.isArray(solution)) {
                solutionArray = solution;
            } else if (typeof solution === 'string') {
                solutionArray = solution.split('');
            } else if (!solution) {
                solutionArray = [];
            } else {
                try {
                    const solutionStr = String(solution);
                    solutionArray = solutionStr.split('');
                } catch (e) {
                    console.error("Errore nella conversione della soluzione:", e);
                    solutionArray = [];
                }
            }
            
            console.log("Soluzione finale da visualizzare:", solutionArray);
            
            try {
                // Chiama il server per generare tutti gli stati del gioco
                console.log("Richiesta stati del gioco al server...");
                const response = await fetch('/generate_game_states', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        ascii_map: asciiMap,
                        solution: solutionArray
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`Errore HTTP: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.status !== 'success') {
                    throw new Error(data.error || 'Errore nella generazione degli stati');
                }
                
                console.log(`Stati del gioco ricevuti: ${data.total_steps} stati`);
                console.log(`🔍 DEBUG: Array precomputedStates avrà ${data.states.length} elementi (indici da 0 a ${data.states.length - 1})`);
                
                // Salva gli stati pre-calcolati
                precomputedStates = data.states;
                currentSolution = solutionArray;
                
                // 💾 Salva automaticamente la soluzione in JSON se c'è una soluzione valida
                if (precomputedStates.length > 1) { // Solo se ci sono effettivamente delle mosse
                    await saveStepByStepSolution(precomputedStates, levelId);
                }
                
                // Visualizza lo stato iniziale
                if (precomputedStates.length > 0) {
                    console.log(`🔍 DEBUG: Mostrando stato iniziale (indice 0). Indice massimo disponibile: ${precomputedStates.length - 1}`);
                    currentStepIndex = 0;
                    await displayPrecomputedStep(0, callbacks, canvas);
                    
                    if (precomputedStates.length > 1 && callbacks.onSolutionReady) {
                        const numberOfMoves = precomputedStates.length - 1;
                        console.log(`🔍 DEBUG: Numero di mosse disponibili: ${numberOfMoves}`);
                        callbacks.onSolutionReady(numberOfMoves);
                    }
                } else {
                    // Fallback: mostra la mappa iniziale
                    const initialMapArray = parseAsciiMap(asciiMap);
                    await drawMap(initialMapArray, canvas);
                    if (callbacks.onNoSolution) {
                        callbacks.onNoSolution();
                    }
                }
                
            } catch (error) {
                console.error("Errore nella generazione degli stati del gioco:", error);
                
                // Fallback: mostra la mappa iniziale senza soluzione
                const initialMapArray = parseAsciiMap(asciiMap);
                await drawMap(initialMapArray, canvas);
                if (callbacks.onNoSolution) {
                    callbacks.onNoSolution();
                }
            }
            
            return this;
        },
        
        /**
         * Visualizza uno step specifico della soluzione
         * @param {number} stepIndex Indice dello step da visualizzare
         * @param {Object} callbacks Funzioni di callback
         * @param {HTMLCanvasElement} canvas Canvas su cui disegnare
         */
        displayStep: async function(stepIndex, callbacks = {}, canvas) {
            await displayPrecomputedStep(stepIndex, callbacks, canvas);
            return this;
        },
        
        /**
         * Avvia o mette in pausa l'animazione della soluzione
         * @param {Object} callbacks Funzioni di callback
         * @param {HTMLCanvasElement} canvas Canvas su cui disegnare
         * @param {number} speed Velocità dell'animazione in ms
         * @returns {boolean} true se l'animazione è stata avviata, false se è stata messa in pausa
         */
        toggleSolutionPlayback: function(callbacks = {}, canvas, speed = 300) {
            if (solutionPlayerInterval) {
                clearInterval(solutionPlayerInterval);
                solutionPlayerInterval = null;
                return false; // Animazione messa in pausa
            } else {
                // Il numero massimo di indici disponibili è precomputedStates.length - 1
                const maxStepIndex = precomputedStates && precomputedStates.length > 0 
                    ? precomputedStates.length - 1  // L'ultimo indice valido
                    : currentSolution.length - 1;
                
                console.log(`🔍 DEBUG: toggleSolutionPlayback - maxStepIndex: ${maxStepIndex}, currentStepIndex: ${currentStepIndex}`);
                    
                if (maxStepIndex < 0) return false;
                
                // Se siamo già all'ultimo step, ricomincia dall'inizio
                if (currentStepIndex >= maxStepIndex) {
                    console.log(`🔍 DEBUG: Siamo all'ultimo step (${currentStepIndex} >= ${maxStepIndex}), ricomincio dall'inizio`);
                    currentStepIndex = -1;
                }
                
                solutionPlayerInterval = setInterval(() => {
                    console.log(`🔍 DEBUG: Interval - currentStepIndex: ${currentStepIndex}, maxStepIndex: ${maxStepIndex}`);
                    if (currentStepIndex < maxStepIndex) {
                        const nextStep = currentStepIndex + 1;
                        console.log(`🔍 DEBUG: Prossimo step: ${nextStep}`);
                        displayPrecomputedStep(nextStep, callbacks, canvas);
                    } else {
                        console.log(`🔍 DEBUG: Animazione completata, fermando interval`);
                        clearInterval(solutionPlayerInterval);
                        solutionPlayerInterval = null;
                        if (callbacks.onPlaybackEnd) {
                            callbacks.onPlaybackEnd();
                        }
                    }
                }, speed);
                
                return true; // Animazione avviata
            }
        },
        
        /**
         * Pulisce e resetta lo stato del modulo
         */
        cleanup: function() {
            if (solutionPlayerInterval) {
                clearInterval(solutionPlayerInterval);
                solutionPlayerInterval = null;
            }
            currentSolution = [];
            currentStepIndex = -1;
            precomputedStates = []; // Reset degli stati pre-calcolati
            
            return this;
        },
        
        /**
         * Restituisce lo stato corrente dell'animazione
         */
        getState: function() {
            // Il numero di mosse è precomputedStates.length - 1 (escludendo lo stato iniziale)
            // Ma il totalSteps per la UI dovrebbe essere il numero massimo di step raggiungibili
            const totalSteps = precomputedStates && precomputedStates.length > 0 
                ? precomputedStates.length - 1  // Numero di mosse (escluso stato iniziale)
                : currentSolution.length;
                
            return {
                isPlaying: solutionPlayerInterval !== null,
                currentStep: currentStepIndex,
                totalSteps: totalSteps,
                hasSolution: totalSteps > 0
            };
        },
        
        /**
         * Disegna una mappa vuota o un messaggio sul canvas
         * @param {HTMLCanvasElement} canvas Canvas su cui disegnare
         */
        drawEmptyMap: function(canvas) {
            if (!ctx) {
                ctx = canvas.getContext('2d');
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#333';
            ctx.textAlign = 'center';
            ctx.fillText("Mappa non disponibile o vuota.", canvas.width / 2, canvas.height / 2);
            
            return this;
        }
    };
})();

// Esporta il modulo
window.GameVisualization = GameVisualization;
