/**
 * Utility: debounce a function call.
 */
function _debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

class MeteoMapManager {
    constructor() {
        this.map = null;
        this.currentGeoJsonLayer = null;
        this.currentValueData = null;
        this.gridLayers = {};
        this._jsonCache = new Map(); // In-memory cache for JSON fetches

        // Initialize Web Workers for off-main-thread processing
        this._colorWorker = null;
        this._jsonWorker = null;
        this._jsonWorkerCallbacks = new Map();
        this._jsonWorkerId = 0;
        try {
            this._colorWorker = new Worker('assets/js/workers/color-calc.worker.js');
            this._jsonWorker = new Worker('assets/js/workers/json-parser.worker.js');
            this._jsonWorker.onmessage = (e) => {
                const { id, data, error } = e.data;
                const cb = this._jsonWorkerCallbacks.get(id);
                if (cb) {
                    this._jsonWorkerCallbacks.delete(id);
                    if (error) cb.reject(new Error(error));
                    else cb.resolve(data);
                }
            };
        } catch (err) {
            console.warn('Web Workers not available, falling back to main thread:', err);
        }
        this.esGeoJson = null;
        this.esGrid = null;
        this.selectedMarker = null; // Marcador visual da célula selecionada
        this.customParameters = {}; // Inicializar parâmetros customizados
        this.windHeight = 50; // Altura padrão para eólico

        this.state = {
            type: 'solar',
            index: 7,
            isPlaying: false,
            isLooping: false,
            isClippedToEs: false,
            maxLayer: 73,
            initialDateTime: null,
            initialIndex: null,
            dateTimePattern: null,
            intervalId: null,
            selectedCell: null
        };

        this.initMap();
        this.setupEventListeners();
        this.setupDomainIndicators();
        this.loadEsGeoJson();
        this.loadCustomParameters();
    }

    /**
     * Fetch JSON with in-memory cache.
     * Avoids re-downloading the same JSON when switching variables or time steps.
     */
    _cachedFetch(url) {
        if (this._jsonCache.has(url)) {
            return Promise.resolve(this._jsonCache.get(url));
        }

        // Use JSON worker if available (parses JSON off main thread)
        const fetchPromise = this._jsonWorker
            ? this._workerFetch(url)
            : fetch(url).then(res => {
                if (!res.ok) throw new Error('Dados não encontrados');
                return res.json();
            });

        return fetchPromise.then(data => {
            // Keep cache bounded (max ~200 entries)
            if (this._jsonCache.size > 200) {
                const firstKey = this._jsonCache.keys().next().value;
                this._jsonCache.delete(firstKey);
            }
            this._jsonCache.set(url, data);
            return data;
        });
    }

    /**
     * Fetch + parse JSON using the Web Worker.
     * Returns a Promise that resolves with the parsed JSON.
     */
    _workerFetch(url) {
        return new Promise((resolve, reject) => {
            const id = String(++this._jsonWorkerId);
            this._jsonWorkerCallbacks.set(id, { resolve, reject });
            // Workers resolve relative URLs from their own location (js/workers/),
            // so we must pass the full absolute URL.
            const absoluteUrl = new URL(url, window.location.href).href;
            this._jsonWorker.postMessage({ url: absoluteUrl, id });
        });
    }

    getVariableId(variableType) {
        // Retorna o ID correto da variável, considerando altura para eólico
        const config = VARIABLES_CONFIG[variableType];
        if (!config) return null;

        if (variableType === 'eolico') {
            if (this.windHeight === 100) return config.id_100m;
            if (this.windHeight === 150) return config.id_150m;
            return config.id;
        }

        return config.id;
    }

    setWindHeight(height) {
        // Muda a altura e recarrega o mapa
        if ([50, 100, 150].includes(height)) {
            this.windHeight = height;
            if (this.state.type === 'eolico') {
                // Limpar cache da grade de camadas para forçar recarregamento
                this.gridLayers = {};

                // Se há uma célula selecionada, fazer re-click para atualizar dados da nova altura
                if (this.state.selectedCell) {
                    this.applyMapChanges().then(() => {
                        // Re-executar o clique nas mesmas coordenadas para atualizar dados da nova altura
                        this.handleMapClick({ latlng: L.latLng(this.state.selectedCell.lat, this.state.selectedCell.lng) });
                    });
                } else {
                    // Sem célula selecionada, apenas recarregar o mapa
                    this.applyMapChanges();
                }
            }
        }
    }

    // ===== GERENCIAMENTO DE PARÂMETROS CUSTOMIZADOS =====

    loadCustomParameters() {
        try {
            const saved = localStorage.getItem('meteoMapCustomParameters');
            this.customParameters = saved ? JSON.parse(saved) : {};
        } catch (e) {
            console.warn('Erro ao carregar parâmetros customizados:', e);
            this.customParameters = {};
        }
    }

    getCustomParameter(variableType, paramName) {
        if (!this.customParameters) {
            this.customParameters = {};
            return null;
        }

        const key = `${variableType}_${paramName}`;
        const customValue = this.customParameters[key];

        // Se houver valor customizado e não for nulo/undefined, usar
        if (customValue !== undefined && customValue !== null && customValue !== '') {
            const numValue = parseFloat(customValue);
            if (!isNaN(numValue)) {
                console.log(`[PARAM GET] ${key} = ${numValue} (customizado)`);
                return numValue;
            }
        }

        console.log(`[PARAM GET] ${key} = null (usando padrão)`);
        return null; // Retorna null para usar o valor padrão do config
    }

    setCustomParameter(variableType, paramName, value) {
        if (!this.customParameters) {
            this.customParameters = {};
        }

        const key = `${variableType}_${paramName}`;

        // Se vazio, null ou undefined, remover do customParameters
        if (value === null || value === undefined || value === '') {
            console.log(`[PARAM SAVE] Removendo ${key} do customParameters`);
            delete this.customParameters[key];
        } else {
            // Validar se é número
            const numValue = parseFloat(value);
            if (!isNaN(numValue)) {
                console.log(`[PARAM SAVE] Salvando ${key} = ${numValue}`);
                this.customParameters[key] = numValue;
            } else {
                // Valor inválido, remover
                console.warn(`[PARAM SAVE] Valor inválido para ${key}: "${value}", removendo`);
                delete this.customParameters[key];
            }
        }

        // Salvar no localStorage
        try {
            localStorage.setItem('meteoMapCustomParameters', JSON.stringify(this.customParameters));
            console.log(`[PARAM SAVE] customParameters atual:`, this.customParameters);

            // Recarregar gráficos se existem
            if (typeof chartsManager !== 'undefined' && chartsManager) {
                chartsManager.reloadChartsWithNewParameters();
            }
        } catch (e) {
            console.warn('Erro ao salvar parâmetros no localStorage:', e);
        }
    }

    resetCustomParameters(variableType) {
        if (!this.customParameters) {
            this.customParameters = {};
            return;
        }

        const prefix = `${variableType}_`;
        Object.keys(this.customParameters).forEach(key => {
            if (key.startsWith(prefix)) {
                delete this.customParameters[key];
            }
        });

        // Salvar no localStorage
        try {
            localStorage.setItem('meteoMapCustomParameters', JSON.stringify(this.customParameters));

            // Recarregar gráficos se existem
            if (typeof chartsManager !== 'undefined' && chartsManager) {
                chartsManager.reloadChartsWithNewParameters();
            }
        } catch (e) {
            console.warn('Erro ao salvar parâmetros no localStorage:', e);
        }
    }

    getEditableParameters(variableType) {
        const params = {
            solar: [
                { name: 'panelEfficiency', label: 'Eficiência do Painel', unit: '%', default: 18 },
                { name: 'inversorEfficiency', label: 'Eficiência do Inversor', unit: '%', default: 95 },
                { name: 'noct', label: 'NOCT', unit: '°C', default: 45 },
                { name: 'ptc', label: 'Coeficiente PTC', unit: '%/°C', default: -0.38 }
            ],
            eolico: [
                { name: 'airDensity', label: 'Densidade do Ar', unit: 'kg/m³', default: 1.225 },
                { name: 'rotorDiameter', label: 'Diâmetro do Rotor', unit: 'm', default: 40 },
                { name: 'Cp', label: 'Coeficiente de Potência da Turbina', unit: '', default: 0.4 }
            ]
        };

        return params[variableType] || [];
    }

    createParametersEditor(variableType) {
        const params = this.getEditableParameters(variableType);

        if (params.length === 0) {
            return '';
        }

        // Garantir que customParameters existe
        if (!this.customParameters) {
            this.customParameters = {};
        }

        let html = `
            <div class="parameters-editor">
                <div class="parameters-toggle" data-variable="${variableType}">
                    <span class="parameters-toggle-label">
                        <i class="fas fa-sliders-h"></i> Parâmetros Customizados
                    </span>
                    <span class="parameters-toggle-icon">▼</span>
                </div>
                <div class="parameters-list" data-variable="${variableType}">
        `;

        params.forEach(param => {
            const customValue = this.customParameters[`${variableType}_${param.name}`];
            const displayValue = customValue !== undefined && customValue !== null ? customValue.toString() : '';

            html += `
                <div class="parameter-item">
                    <label class="parameter-label">${param.label}</label>
                    <input 
                        type="text" 
                        class="parameter-input parameter-${variableType}-${param.name}" 
                        placeholder="${param.default} (padrão)"
                        value="${displayValue}"
                        data-variable="${variableType}"
                        data-param="${param.name}"
                        data-default="${param.default}"
                        inputmode="decimal"
                    />
                    <span class="parameter-unit">${param.unit}</span>
                    <span class="parameter-hint">Deixe em branco para usar padrão</span>
                </div>
            `;
        });

        html += `
                    <button class="reset-parameters-btn" data-variable="${variableType}">
                        <i class="fas fa-redo"></i> Restaurar Padrões
                    </button>
                </div>
            </div>
        `;

        return html;
    }

    setupParametersEditorListeners(variableType) {
        try {
            const toggle = document.querySelector(`.parameters-toggle[data-variable="${variableType}"]`);
            const list = document.querySelector(`.parameters-list[data-variable="${variableType}"]`);
            const resetBtn = document.querySelector(`.reset-parameters-btn[data-variable="${variableType}"]`);
            const inputs = document.querySelectorAll(`[data-variable="${variableType}"].parameter-input`);

            if (toggle && list) {
                toggle.addEventListener('click', () => {
                    const icon = toggle.querySelector('.parameters-toggle-icon');
                    list.classList.toggle('active');
                    icon.classList.toggle('active');
                });
            }

            if (inputs.length > 0) {
                inputs.forEach(input => {
                    // Função para validar e salvar
                    const validateAndSave = (e) => {
                        const value = e.target.value.trim();
                        const paramName = e.target.dataset.param;
                        const defaultValue = e.target.dataset.default;

                        console.log(`[PARAM UPDATE] Validando ${variableType}.${paramName}: "${value}"`);

                        // Se vazio, usar padrão
                        if (value === '') {
                            this.setCustomParameter(variableType, paramName, null);
                            e.target.value = '';
                            console.log(`[PARAM UPDATE] Valor limpo, usando padrão`);
                        } else {
                            // Validar se é número
                            const numValue = parseFloat(value);
                            if (isNaN(numValue)) {
                                // Valor inválido, restaurar padrão
                                console.warn(`[PARAM UPDATE] Valor inválido para ${paramName}: "${value}". Usando padrão.`);
                                e.target.value = '';
                                this.setCustomParameter(variableType, paramName, null);
                            } else {
                                // Valor válido, salvar
                                console.log(`[PARAM UPDATE] Salvando valor válido: ${numValue}`);
                                this.setCustomParameter(variableType, paramName, numValue);
                                e.target.value = numValue.toString();
                            }
                        }

                        // Atualizar a sidebar com os novos cálculos
                        console.log(`[PARAM UPDATE] Atualizando sidebar...`);
                        this.updateSidebarWithNewParameters(variableType);
                    };

                    // Validar ao sair do campo (blur)
                    input.addEventListener('blur', validateAndSave);

                    // Validar ao pressionar Enter
                    input.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter' || e.code === 'Enter') {
                            e.preventDefault();
                            validateAndSave(e);
                            input.blur();
                        }
                    });

                    // Permitir apenas números, ponto e negativos enquanto digita
                    input.addEventListener('input', (e) => {
                        let value = e.target.value;

                        // Remover caracteres inválidos (manter apenas dígitos, ponto e hífen no início)
                        value = value.replace(/[^\d.\-]/g, '');

                        // Evitar múltiplos pontos
                        const parts = value.split('.');
                        if (parts.length > 2) {
                            value = parts[0] + '.' + parts.slice(1).join('');
                        }

                        // Evitar múltiplos hífens
                        if ((value.match(/-/g) || []).length > 1) {
                            value = value.replace(/-/g, '');
                            value = '-' + value;
                        }

                        e.target.value = value;
                    });
                });
            }

            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                    if (confirm('Restaurar parâmetros aos valores padrão?')) {
                        console.log(`[PARAM RESET] Restaurando valores padrão para ${variableType}`);
                        this.resetCustomParameters(variableType);
                        inputs.forEach(input => {
                            input.value = '';
                        });
                        // Atualizar a sidebar com os valores padrão
                        this.updateSidebarWithNewParameters(variableType);
                    }
                });
            }
        } catch (e) {
            console.warn('Erro ao configurar listeners dos parâmetros:', e);
        }
    }

    updateSidebarWithNewParameters(variableType) {
        console.log(`[SIDEBAR UPDATE] Iniciando atualização para ${variableType}`);

        // Atualizar a sidebar com novos cálculos baseado nos parâmetros alterados
        if (!this.state.selectedCell) {
            console.warn(`[SIDEBAR UPDATE] Nenhuma célula selecionada`);
            return;
        }

        if (this.state.type !== variableType) {
            console.warn(`[SIDEBAR UPDATE] Variável mismatch: ${this.state.type} !== ${variableType}`);
            return;
        }

        const config = VARIABLES_CONFIG[this.state.type];
        if (!config || !config.specificInfo) {
            console.warn(`[SIDEBAR UPDATE] Config ou specificInfo não encontrados`);
            return;
        }

        console.log(`[SIDEBAR UPDATE] Recalculando para valor=${this.state.selectedCell.value}, allValues=`, this.state.selectedCell.allValues);

        // Recalcular as informações específicas com os novos parâmetros
        const specificInfo = config.specificInfo(
            this.state.selectedCell.value,
            this.state.selectedCell.allValues
        );

        console.log(`[SIDEBAR UPDATE] Novo specificInfo:`, specificInfo);

        // Atualizar apenas a seção de informações específicas
        this.updateSidebarSpecificInfo(specificInfo);

        console.log(`[SIDEBAR UPDATE] Atualização concluída`);
    }

    updateSidebarSpecificInfo(specificInfo) {
        const sidebarContent = document.getElementById('sidebarContent');
        const existingSpecific = sidebarContent.querySelector('.variable-specific');

        if (!existingSpecific) return;

        // Preservar o estado do dropdown antes de atualizar
        const existingEditor = existingSpecific.querySelector('.parameters-editor');
        let wasEditorOpen = false;
        if (existingEditor) {
            const existingList = existingEditor.querySelector('.parameters-list');
            wasEditorOpen = existingList && existingList.classList.contains('active');
            console.log(`[SIDEBAR UPDATE] Estado do dropdown antes: ${wasEditorOpen ? 'aberto' : 'fechado'}`);
        }

        let html = `
            <div class="info-section-title">
                <i class="fas fa-bolt"></i> ${specificInfo.title}
            </div>
        `;

        specificInfo.items.forEach(item => {
            html += `
                <div class="stat-card">
                    <div style="color: #666; font-size: 0.85rem;">
                        <i class="fas ${item.icon}"></i> ${item.label}
                    </div>
                    <div class="stat-card-value">
                        ${item.value}
                        <span class="stat-card-unit">${item.unit || ''}</span>
                    </div>
                </div>
            `;
        });

        // Reconstruir o editor de parâmetros com HTML novo
        const editorHTML = this.createParametersEditor(this.state.type);
        html += editorHTML;

        existingSpecific.innerHTML = html;

        // Restaurar o estado do dropdown se estava aberto
        if (wasEditorOpen) {
            const newList = existingSpecific.querySelector('.parameters-list');
            const newToggle = existingSpecific.querySelector('.parameters-toggle');
            if (newList && newToggle) {
                newList.classList.add('active');
                const icon = newToggle.querySelector('.parameters-toggle-icon');
                if (icon) icon.classList.add('active');
                console.log(`[SIDEBAR UPDATE] Dropdown restaurado para aberto`);
            }
        }

        // Reconfigurar listeners para o editor de parâmetros
        this.setupParametersEditorListeners(this.state.type);
    }

    initMap() {
        // Canvas renderer: toda a grade fica em <canvas> em vez de SVG individual
        // Elimina reflows causados por mutar atributos de paths SVG (setStyle por lãyer)
        this._canvasRenderer = L.canvas({ padding: 0.5 });

        this.map = L.map('map', {
            fadeAnimation: true,
            maxZoom: 15,
            renderer: this._canvasRenderer
        }).setView([-12.97, -38.5], 6);

        L.tileLayer('https://{s}.tile.osm.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap | LabMiM-UFBA',
            minZoom: 3,
            maxZoom: 15
        }).addTo(this.map);

        // Setup canvas para vetores de vento
        this.setupWindCanvas();
    }

    setupWindCanvas() {
        const canvas = document.getElementById('windVectorCanvas');
        if (canvas) {
            // Armazenar função nomeada para poder remover listener específico
            if (!this.windCanvasUpdateHandler) {
                this.windCanvasUpdateHandler = () => {
                    canvas.width = this.map.getSize().x;
                    canvas.height = this.map.getSize().y;

                    // Re-render vetores se estiverem ativos
                    const windCheckbox = document.getElementById('windLayerCheckbox');
                    if (windCheckbox && windCheckbox.checked) {
                        // Usar requestAnimationFrame para evitar múltiplas renderizações
                        cancelAnimationFrame(this.windRenderScheduled);
                        this.windRenderScheduled = requestAnimationFrame(() => this.renderWindVectors());
                    }
                };

                // Executar inicialmente
                this.windCanvasUpdateHandler();

                // Anexar listeners específicos para vento
                this.map.on('move', this.windCanvasUpdateHandler, this);
                this.map.on('resize', this.windCanvasUpdateHandler, this);
                this.map.on('zoomend', this.windCanvasUpdateHandler, this);
            }
        }
    }

    setupEventListeners() {
        const slider = document.getElementById('layerSlider');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const loopToggleBtn = document.getElementById('loopToggleBtn');
        const clipToEsBtn = document.getElementById('clipToEsBtn');
        const variableSelect = document.getElementById('variableSelect');
        const closeSidebarBtn = document.getElementById('closeSidebarBtn');

        // Debounced handler for slider to avoid avalanche of requests
        const _debouncedSliderApply = _debounce(() => {
            if (this.state.selectedCell && !this.state.isPlaying) {
                this.applyMapChanges().then(() => {
                    this.handleMapClick({ latlng: L.latLng(this.state.selectedCell.lat, this.state.selectedCell.lng) });
                });
            } else {
                this.applyMapChanges();
            }
        }, 100);

        slider.addEventListener('input', (e) => {
            this.state.index = parseInt(e.target.value);
            this.updateDateTime();
            _debouncedSliderApply();
        });

        playPauseBtn.addEventListener('click', () => {
            // Fechar sidebar ao começar animação
            if (!this.state.isPlaying) {
                this.closeSidebar();
            }
            this.togglePlayPause();
        });
        loopToggleBtn.addEventListener('click', () => this.toggleLoop());
        clipToEsBtn.addEventListener('click', () => this.toggleClipToEs(clipToEsBtn));
        variableSelect.addEventListener('change', (e) => this.switchVariable(e.target.value));
        closeSidebarBtn.addEventListener('click', () => this.closeSidebar());

        // Event listeners para seletor de altura (eólico)
        const heightButtons = document.querySelectorAll('.height-btn');
        heightButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const height = parseInt(e.target.dataset.height);
                heightButtons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.setWindHeight(height);
            });
        });

        // Event listeners para camada de vetores de vento
        const windCheckbox = document.getElementById('windLayerCheckbox');
        if (windCheckbox) {
            windCheckbox.addEventListener('change', (e) => {
                this.toggleWindLayer(e.target.checked);
            });
        }

        this.map.on('click', (e) => this.handleMapClick(e));
        this.map.on('zoomend', () => {
            this.updateDomainIndicator();
            this.applyMapChanges();
        });

        // Atualizar domínio inicial
        this.updateDomainIndicator();

        // Documentation modal event listeners
        this.setupDocumentationListeners();
    }

    setupDocumentationListeners() {
        const docBtn = document.getElementById('docBtn');
        const docCloseBtn = document.getElementById('docCloseBtn');
        const docModal = document.getElementById('documentationModal');
        const docTabs = document.querySelectorAll('.doc-tab');

        // Abrir modal
        docBtn.addEventListener('click', () => {
            docModal.classList.add('active');
        });

        // Fechar modal com botão X
        docCloseBtn.addEventListener('click', () => {
            docModal.classList.remove('active');
        });

        // Fechar modal clicando fora do conteúdo
        docModal.addEventListener('click', (e) => {
            if (e.target === docModal) {
                docModal.classList.remove('active');
            }
        });

        // Gerenciar abas
        docTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');

                // Remover active de todas as abas e conteúdos
                docTabs.forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.doc-tab-content').forEach(content => {
                    content.classList.remove('active');
                });

                // Adicionar active à aba clicada e seu conteúdo
                tab.classList.add('active');
                document.querySelector(`.doc-tab-content[data-tab="${tabName}"]`).classList.add('active');
            });
        });

        // Fechar com ESC
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && docModal.classList.contains('active')) {
                docModal.classList.remove('active');
            }
        });
    }

    loadEsGeoJson() {
        fetch('https://raw.githubusercontent.com/giuliano-macedo/geodata-br-states/main/geojson/br_states/br_ba.json')
            .then(res => res.json())
            .then(geojson => {
                this.esGeoJson = geojson;
                const bbox = turf.bbox(geojson.features[0]);
                const grid = turf.squareGrid(bbox, 9, { units: 'kilometers' });
                this.esGrid = grid.features.filter(cell =>
                    turf.booleanIntersects(cell, geojson.features[0])
                );
                this.applyMapChanges();
            })
            .catch(err => console.error('Erro ao carregar BA:', err));
    }

    updateDateTime() {
        const config = VARIABLES_CONFIG[this.state.type];
        const hour = ((this.state.index - 1) % 24);

        if (config.id === 'SWDOWN' && (hour < 6 || hour > 18)) {
            document.getElementById('layerLabel').textContent = 'Sem dados (noturno)';
        } else {
            const label = this.calculateDateTimeFromIndex(this.state.index);
            document.getElementById('layerLabel').textContent = label;
        }
    }

    calculateDateTimeFromIndex(index) {
        if (!this.state.initialDateTime) return `Hora ${index}`;

        const hoursDiff = index - this.state.initialIndex;
        const date = new Date(this.state.initialDateTime);
        date.setHours(date.getHours() + hoursDiff);

        return date.toLocaleString('pt-BR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    togglePlayPause() {
        this.state.isPlaying = !this.state.isPlaying;
        const btn = document.getElementById('playPauseBtn');

        if (this.state.isPlaying) {
            btn.innerHTML = '<i class="fas fa-pause"></i> Pause';
            this.startAnimation();
        } else {
            btn.innerHTML = '<i class="fas fa-play"></i> Play';
            this.stopAnimation();
        }
    }

    toggleLoop() {
        this.state.isLooping = !this.state.isLooping;
        const btn = document.getElementById('loopToggleBtn');
        btn.textContent = this.state.isLooping ? '🔁 Loop On' : '🔁 Loop Off';
        btn.classList.toggle('active', this.state.isLooping);
    }

    toggleClipToEs(btn) {
        this.state.isClippedToEs = !this.state.isClippedToEs;
        btn.innerHTML = this.state.isClippedToEs ?
            '<i class="fas fa-map"></i> BA On' :
            '<i class="fas fa-map"></i> BA Off';
        btn.classList.toggle('active', this.state.isClippedToEs);
        this.applyMapChanges();
    }

    startAnimation() {
        this.state.intervalId = setInterval(() => {
            const slider = document.getElementById('layerSlider');
            let nextIndex = parseInt(slider.value) + 1;

            const config = VARIABLES_CONFIG[this.state.type];
            const nextHour = ((nextIndex - 1) % 24);

            if (config.id === 'SWDOWN' && (nextHour < 6 || nextHour > 18)) {
                nextIndex = nextHour < 6 ?
                    Math.floor(nextIndex / 24) * 24 + 7 :
                    Math.ceil(nextIndex / 24) * 24 + 7;
            }

            if (nextIndex > this.state.maxLayer) {
                nextIndex = this.state.isLooping ?
                    (config.id === 'SWDOWN' ? 7 : 1) :
                    this.state.maxLayer;

                if (!this.state.isLooping) this.stopAnimation();
            }

            slider.value = nextIndex;
            slider.dispatchEvent(new Event('input'));
        }, 800);
    }

    stopAnimation() {
        clearInterval(this.state.intervalId);
        this.state.isPlaying = false;
        document.getElementById('playPauseBtn').innerHTML = '<i class="fas fa-play"></i> Play';
    }

    switchVariable(variableType) {
        this.gridLayers = {};
        this.state.type = variableType;
        document.getElementById('variableSelect').value = variableType;

        // Mostrar/esconder seletor de altura
        const heightSelector = document.getElementById('heightSelector');
        const windLayerToggle = document.getElementById('windLayerToggle');
        const windVectorLegend = document.getElementById('windVectorLegend');

        // IMPORTANTE: Sempre limpar canvas de vento ao trocar de variável
        const windCanvas = document.getElementById('windVectorCanvas');
        if (windCanvas) {
            const ctx = windCanvas.getContext('2d');
            ctx.clearRect(0, 0, windCanvas.width, windCanvas.height);
        }

        if (variableType === 'eolico') {
            heightSelector.classList.add('active');
            if (windLayerToggle) windLayerToggle.classList.add('active');
        } else {
            heightSelector.classList.remove('active');
            if (windLayerToggle) windLayerToggle.classList.remove('active');
            // Desativar camada de vento se estava ativa e limpar canvas
            const windCheckbox = document.getElementById('windLayerCheckbox');
            if (windCheckbox && windCheckbox.checked) {
                windCheckbox.checked = false;
                this.toggleWindLayer(false);
            }
        }

        // Armazenar coordenadas da célula selecionada antes de remover o marcador
        const selectedCellCoords = this.state.selectedCell ? {
            lat: this.state.selectedCell.lat,
            lng: this.state.selectedCell.lng
        } : null;

        // Remover marcador anterior se existir (antes de carregar dados)
        if (this.selectedMarker) {
            this.map.removeLayer(this.selectedMarker);
            this.selectedMarker = null;
        }

        // IMPORTANTE: Chamar applyMapChanges() ANTES de updateSelectedCellData()
        // para que this.currentValueData seja atualizado com a nova variável
        this.applyMapChanges().then(() => {
            // Agora this.currentValueData tem os dados corretos da nova variável
            if (this.state.selectedCell && selectedCellCoords) {
                // Re-executar o clique nas mesmas coordenadas para atualizar dados da nova variável
                this.handleMapClick({ latlng: L.latLng(selectedCellCoords.lat, selectedCellCoords.lng) });
            } else if (this.state.selectedCell) {
                this.updateSelectedCellData();
            }
        });
    }

    applyMapChanges() {
        const zoom = this.map.getZoom();
        const config = VARIABLES_CONFIG[this.state.type];
        const hour = ((this.state.index - 1) % 24);

        if (zoom < 5) {
            this.removeCurrentLayer();
            // Remover marcador ao mudar de domínio
            if (this.selectedMarker) {
                this.map.removeLayer(this.selectedMarker);
                this.selectedMarker = null;
            }
            this.updateDateTime();
            return Promise.resolve();
        }

        if (config.id === 'SWDOWN' && (hour < 6 || hour > 18)) {
            this.removeCurrentLayer();
            // Remover marcador ao mudar domínio (noturno)
            if (this.selectedMarker) {
                this.map.removeLayer(this.selectedMarker);
                this.selectedMarker = null;
            }
            this.updateDateTime();
            return Promise.resolve();
        }

        return this.loadValueData(this.state.index, this.state.type);
    }

    loadValueData(index, type) {
        const config = VARIABLES_CONFIG[type];
        const zoom = this.map.getZoom();
        let domain = this.getDomainFromZoom(zoom);

        if (!domain) {
            this.removeCurrentLayer();
            return Promise.resolve(null);
        }

        const id_num = String(index).padStart(3, '0');
        const variableId = this.getVariableId(type);
        const filePath = `JSON/${domain}_${variableId}_${id_num}.json`;

        return this._cachedFetch(filePath)
            .then(valueData => this.loadGridLayer(domain, type).then(gridLayer => {
                if (!gridLayer) return null;

                this.applyValuesToGrid(gridLayer, valueData);

                let displayLayer = gridLayer;
                if (this.state.isClippedToEs) {
                    displayLayer = this.filterCellsToEs(gridLayer);
                }

                this.showGeoJsonLayer(displayLayer);
                this.currentValueData = valueData;
                this.updateUIFromMetadata(valueData.metadata, gridLayer._gridMetadata);

                // Re-render vetores de vento se habilitados (para eólico)
                if (type === 'eolico') {
                    const windCheckbox = document.getElementById('windLayerCheckbox');
                    if (windCheckbox && windCheckbox.checked) {
                        // Usar setTimeout para garantir que o canvas está pronto
                        setTimeout(() => this.renderWindVectors(), 100);
                    }
                }

                return valueData;
            }))
            .catch(err => {
                console.error('Erro ao carregar dados:', err);
                this.removeCurrentLayer();
                return null;
            });
    }

    getDomainFromZoom(zoom) {
        if (zoom >= 5 && zoom <= 6) return 'D01';
        if (zoom >= 7 && zoom <= 8) return 'D02';
        if (zoom >= 9 && zoom <= 11) return 'D03';
        if (zoom >= 12) return 'D04';
        return null;
    }

    updateDomainIndicator() {
        const zoom = this.map.getZoom();
        const domain = this.getDomainFromZoom(zoom);
        const domainButtons = document.querySelectorAll('.domain-btn');

        // Remover active de todos os botões
        domainButtons.forEach(btn => btn.classList.remove('active'));

        // Adicionar active ao botão do domínio atual
        if (domain) {
            const activeBtn = document.querySelector(`.domain-btn[data-domain="${domain}"]`);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
        }
    }

    setupDomainIndicators() {
        // Centros dos domínios (latitude, longitude) e zoom levels
        const domainConfig = {
            'D01': { center: [-12.97, -38.5], zoom: 5.5 },
            'D02': { center: [-12.97, -38.5], zoom: 7 },
            'D03': { center: [-12.97, -38.5], zoom: 9 },
            'D04': { center: [-12.97, -38.5], zoom: 12 }
        };

        const domainButtons = document.querySelectorAll('.domain-btn');

        domainButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const selectedDomain = button.dataset.domain;
                const config = domainConfig[selectedDomain];
                const targetZoom = config.zoom;
                const centerCoords = config.center;

                // Se há célula selecionada, fazer zoom para ela e recarregar dados
                if (this.state.selectedCell) {
                    const selectedLat = this.state.selectedCell.lat;
                    const selectedLng = this.state.selectedCell.lng;

                    // Zoom animado para a célula selecionada
                    this.map.flyTo(
                        [selectedLat, selectedLng],
                        targetZoom,
                        {
                            duration: 1.5,
                            easeLinearity: 0.25
                        }
                    );

                    // Após o zoom completar, recarregar dados e fazer clique
                    this.map.once('moveend', () => {
                        // Recarregar dados do novo domínio e depois fazer o clique
                        this.applyMapChanges().then(() => {
                            this.handleMapClick({
                                latlng: L.latLng(selectedLat, selectedLng)
                            }).catch(() => {
                                // Se não encontrar dados, fechar sidebar
                                this.closeSidebar();
                            });
                        });
                    });
                } else {
                    // Se não há célula selecionada, fazer zoom no centro do domínio
                    this.map.flyTo(
                        centerCoords,
                        targetZoom,
                        {
                            duration: 1.5,
                            easeLinearity: 0.25
                        }
                    );
                }
            });
        });
    }

    loadGridLayer(domain, type) {
        // Grid geometry is identical for all variables in a domain,
        // so we cache by domain only (not domain+variable)
        const cacheKey = domain;

        if (this.gridLayers[cacheKey]) {
            return Promise.resolve(this.gridLayers[cacheKey]);
        }

        // Fetch domain-only GeoJSON (e.g. D01.geojson, not D01_TEMP.geojson)
        return fetch(`geoJSON/${domain}.geojson`)
            .then(res => res.json())
            .then(geojson => {
                const gridMetadata = geojson.metadata;
                const layer = L.geoJSON(geojson, {
                    // Usar o mesmo canvas renderer do mapa para evitar SVG reflows
                    renderer: this._canvasRenderer,
                    style: {
                        weight: 0.3,
                        opacity: 0.15,
                        color: 'white',
                        fillColor: '#cccccc',
                        fillOpacity: 0.45
                    },
                    onEachFeature: (feature, layer) => {
                        feature.properties.valor = null;
                        layer.on({
                            mouseover: () => {
                                layer.setStyle({
                                    weight: 1.2,
                                    color: '#666',
                                    fillOpacity: 0.65
                                });
                            },
                            mouseout: () => {
                                layer.setStyle({
                                    weight: 0.3,
                                    color: 'white',
                                    fillOpacity: 0.45
                                });
                            }
                        });
                    }
                });

                layer._gridMetadata = gridMetadata;
                this.gridLayers[cacheKey] = layer;
                return layer;
            })
            .catch(err => {
                console.error('Erro ao carregar grid:', err);
                return null;
            });
    }

    applyValuesToGrid(gridLayer, valueData) {
        const values = valueData.values;
        const layers = gridLayer.getLayers();
        const config = VARIABLES_CONFIG[this.state.type];

        // Pre-compute scale once
        let scaleValues = valueData.metadata.scale_values;
        if (config.useDynamicScale && this.currentValueData) {
            const dynamicScale = this.calculateDynamicScale(this.currentValueData, config);
            if (dynamicScale) scaleValues = dynamicScale;
        }

        // Offload color computation to Web Worker if available
        if (this._colorWorker) {
            this._colorWorker.onmessage = (e) => {
                const { colors } = e.data;
                cancelAnimationFrame(this._applyGridRaf);
                this._applyGridRaf = requestAnimationFrame(() => {
                    for (let i = 0; i < layers.length; i++) {
                        if (colors[i] !== undefined) {
                            layers[i].feature.properties.valor = values[i];
                            layers[i].setStyle({
                                fillColor: colors[i],
                                fillOpacity: 0.45,
                                weight: 0.5,
                                opacity: 0.15,
                                color: 'white'
                            });
                        }
                    }
                });
            };
            this._colorWorker.postMessage({
                values,
                scaleValues,
                colors: config.colors
            });
        } else {
            // Fallback: compute colors on main thread
            const colors = new Array(layers.length);
            for (let i = 0; i < layers.length; i++) {
                const value = values[i];
                if (value !== undefined && value !== null) {
                    colors[i] = this._colorFromScale(value, scaleValues, config);
                }
            }

            cancelAnimationFrame(this._applyGridRaf);
            this._applyGridRaf = requestAnimationFrame(() => {
                for (let i = 0; i < layers.length; i++) {
                    if (colors[i] !== undefined) {
                        layers[i].feature.properties.valor = values[i];
                        layers[i].setStyle({
                            fillColor: colors[i],
                            fillOpacity: 0.45,
                            weight: 0.5,
                            opacity: 0.15,
                            color: 'white'
                        });
                    }
                }
            });
        }
    }

    // Versão "escalar por lookup" de getColorForValue — recebe scale pré-computado
    _colorFromScale(value, scaleValues, config) {
        if (value < scaleValues[0]) return config.colors[0];
        if (value > scaleValues[scaleValues.length - 1]) return config.colors[config.colors.length - 1];
        for (let i = 0; i < scaleValues.length - 1; i++) {
            if (value >= scaleValues[i] && value < scaleValues[i + 1]) {
                const ratio = (value - scaleValues[i]) / (scaleValues[i + 1] - scaleValues[i]);
                return this.interpolateColor(config.colors, (i + ratio) / (scaleValues.length - 1));
            }
        }
        return config.colors[config.colors.length - 1];
    }

    getColorForValue(value, metadata, config) {
        let scaleValues = metadata.scale_values;

        // Se a variável usa escala dinâmica, ajustar escala com base nos dados atuais
        if (config.useDynamicScale && this.currentValueData) {
            const dynamicScale = this.calculateDynamicScale(this.currentValueData, config);
            if (dynamicScale) {
                scaleValues = dynamicScale;
            }
        }

        if (value < scaleValues[0]) return config.colors[0];
        if (value > scaleValues[scaleValues.length - 1]) return config.colors[config.colors.length - 1];

        for (let i = 0; i < scaleValues.length - 1; i++) {
            if (value >= scaleValues[i] && value < scaleValues[i + 1]) {
                const ratio = (value - scaleValues[i]) / (scaleValues[i + 1] - scaleValues[i]);
                return this.interpolateColor(config.colors, (i + ratio) / (scaleValues.length - 1));
            }
        }

        return config.colors[config.colors.length - 1];
    }

    calculateDynamicScale(valueData, config) {
        // Calcular min/max dos dados para criar uma escala dinâmica
        if (!valueData.features || valueData.features.length === 0) return null;

        let min = Infinity;
        let max = -Infinity;

        valueData.features.forEach(feature => {
            if (feature.properties && feature.properties.value !== null && feature.properties.value !== undefined) {
                const val = feature.properties.value;
                if (val < min) min = val;
                if (val > max) max = val;
            }
        });

        if (min === Infinity || max === -Infinity) return null;

        // Para pressão, usar a média (1013 hPa) como ponto central
        let center = config.normalValue || ((min + max) / 2);
        let range = Math.max(Math.abs(max - center), Math.abs(min - center));

        // Criar escala simétrica em torno do valor normal
        const scaleMin = center - range;
        const scaleMax = center + range;

        // Gerar 10 valores interpolados (mesma quantidade que a paleta de cores)
        const dynamicScale = [];
        for (let i = 0; i < 10; i++) {
            dynamicScale.push(scaleMin + (scaleMax - scaleMin) * (i / 9));
        }

        return dynamicScale;
    }

    interpolateColor(colors, factor) {
        const index = factor * (colors.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const localFactor = index - lower;

        if (lower === upper) return colors[lower];

        const c1 = this.hexToRgb(colors[lower]);
        const c2 = this.hexToRgb(colors[upper]);

        return `rgb(${Math.round(c1.r + (c2.r - c1.r) * localFactor)}, ${Math.round(c1.g + (c2.g - c1.g) * localFactor)}, ${Math.round(c1.b + (c2.b - c1.b) * localFactor)})`;
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    filterCellsToEs(gridLayer) {
        if (!this.esGrid) return gridLayer;

        const filtered = [];
        gridLayer.eachLayer(layer => {
            const bounds = layer.getBounds();
            const center = turf.point([(bounds.getEast() + bounds.getWest()) / 2, (bounds.getNorth() + bounds.getSouth()) / 2]);

            if (this.esGrid.some(cell => turf.booleanPointInPolygon(center, cell))) {
                filtered.push(layer);
            }
        });

        const newLayer = L.layerGroup(filtered);
        newLayer._gridMetadata = gridLayer._gridMetadata;
        return newLayer;
    }

    showGeoJsonLayer(newLayer) {
        if (this.currentGeoJsonLayer) {
            this.map.removeLayer(this.currentGeoJsonLayer);
        }
        newLayer.addTo(this.map);
        this.currentGeoJsonLayer = newLayer;

        // IMPORTANTE: Recriar marcador APÓS o layer ser adicionado
        // Garante que o marcador sempre fica acima do layer
        if (this.state.selectedCell) {
            // Remover marcador antigo
            if (this.selectedMarker) {
                this.map.removeLayer(this.selectedMarker);
                this.selectedMarker = null;
            }
            // Recrear marcador (renderizado por último, fica no topo)
            this.selectedMarker = this.createPingMarker(this.state.selectedCell.lat, this.state.selectedCell.lng);
        }
    }

    removeCurrentLayer() {
        if (this.currentGeoJsonLayer) {
            this.map.removeLayer(this.currentGeoJsonLayer);
            this.currentGeoJsonLayer = null;
        }
    }

    updateUIFromMetadata(metadata, gridMetadata) {
        const config = VARIABLES_CONFIG[this.state.type];

        if (!this.state.initialDateTime) {
            this.state.initialDateTime = this.parseDateTime(metadata.date_time);
            this.state.initialIndex = this.state.index;
        }

        this.updateColorbar(config);
        this.updateDateTime();
    }

    parseDateTime(dateStr) {
        const parts = dateStr.split(' ');
        const dateParts = parts[0].includes('/') ? parts[0].split('/').reverse().join('-') : parts[0];
        return new Date(dateParts + ' ' + parts[1]);
    }

    updateColorbar(config) {
        const gradient = `linear-gradient(to top, ${config.colors.join(', ')})`;
        document.getElementById('colorbarGradient').style.background = gradient;
        document.getElementById('colorbarUnit').textContent = config.unit;

        let scaleValues = this.currentValueData?.metadata.scale_values || [];

        // Se usar escala dinâmica, calcular valores para a colorbar
        if (config.useDynamicScale && this.currentValueData) {
            const dynamicScale = this.calculateDynamicScale(this.currentValueData, config);
            if (dynamicScale) {
                scaleValues = dynamicScale;
            }
        }

        const labelsContainer = document.getElementById('colorbarLabels');
        labelsContainer.innerHTML = '';

        for (let i = scaleValues.length - 1; i >= 0; i--) {
            const label = document.createElement('div');
            label.className = 'colorbar-label';
            label.textContent = scaleValues[i].toFixed(0) + (i === scaleValues.length - 1 ? '+' : '');
            labelsContainer.appendChild(label);
        }
    }

    handleMapClick(e) {
        if (!this.currentGeoJsonLayer) {
            return Promise.reject('No GeoJSON layer available');
        }

        let foundCell = null;
        this.currentGeoJsonLayer.eachLayer(layer => {
            if (layer.getBounds().contains(e.latlng)) {
                foundCell = {
                    layer: layer,
                    value: layer.feature.properties.valor,
                    cellIndex: layer.feature.properties.index,
                    lat: e.latlng.lat,
                    lng: e.latlng.lng,
                    allValues: {} // Armazenar valores de todas as variáveis
                };
            }
        });

        if (!foundCell || foundCell.value === null) {
            this.showErrorMessage('Sem informações meteorológicas neste local');
            return Promise.reject('No cell data found at this location');
        }

        // Remover marcador anterior se existir
        if (this.selectedMarker) {
            this.map.removeLayer(this.selectedMarker);
        }

        // Criar marcador visual com estilo "ping"
        this.selectedMarker = this.createPingMarker(e.latlng.lat, e.latlng.lng);

        // Carregar valores de todas as variáveis para este ponto
        return this.loadAllVariableValuesForCell(foundCell).then(allValues => {
            foundCell.allValues = allValues;
            this.state.selectedCell = foundCell;
            this.showSidebar();
            return foundCell;
        }).catch(err => {
            console.error('Erro ao carregar valores:', err);
            this.showErrorMessage('Erro ao carregar informações meteorológicas');
            // Remover marcador em caso de erro
            if (this.selectedMarker) {
                this.map.removeLayer(this.selectedMarker);
                this.selectedMarker = null;
            }
            throw err;
        });
    }

    createPingMarker(lat, lng) {
        /**
         * Cria um marcador pin clássico com pulso SVG
         * Usa L.divIcon para máxima flexibilidade
         * Tamanho fixo em todos os zooms (32px - tamanho D02)
         */
        const iconSize = 32;
        const scaleFactor = 1;

        const pingIcon = L.divIcon({
            className: 'ping-pin',
            html: `
                <svg width="${iconSize}" height="${iconSize}" viewBox="0 0 24 24" style="transform: scale(${scaleFactor});">
                    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"
                          fill="#ff006e" stroke="#ffffff" stroke-width="0.5"/>
                    <circle cx="12" cy="9" r="2.5" fill="white"/>
                </svg>
                <span class="ping-pulse" style="width: ${iconSize}px; height: ${iconSize}px;"></span>
            `,
            iconSize: [iconSize, iconSize],
            iconAnchor: [iconSize / 2, iconSize],
            popupAnchor: [0, -iconSize]
        });

        return L.marker([lat, lng], {
            icon: pingIcon,
            zIndexOffset: 1000 // Garantir que fica no topo
        }).addTo(this.map);
    }

    loadValueDataOnly(index, type) {
        /**
         * Carrega dados de uma variável SEM renderizar o mapa
         * Usado apenas para obter valores para cálculos multivariáveis
         * Sempre retorna uma promise com null em caso de erro
         */
        const config = VARIABLES_CONFIG[type];
        const zoom = this.map.getZoom();
        let domain = this.getDomainFromZoom(zoom);

        if (!domain) {
            return Promise.resolve(null);
        }

        const id_num = String(index).padStart(3, '0');
        const variableId = this.getVariableId(type);
        const filePath = `JSON/${domain}_${variableId}_${id_num}.json`;

        return this._cachedFetch(filePath).catch(() => null);
    }

    loadAllVariableValuesForCell(foundCell) {
        const allValues = {};

        const promises = [];

        // Carregar valores de todas as variáveis para este ponto (SEM renderizar mapa)
        Object.keys(VARIABLES_CONFIG).forEach(varType => {
            const config = VARIABLES_CONFIG[varType];

            // Se a variável é a que está sendo exibida no mapa, usar o valor da célula clicada
            if (varType === this.state.type && foundCell) {
                allValues[varType] = {
                    value: foundCell.value,
                    label: config.label,
                    unit: config.unit
                };
                return; // Não precisa carregar do JSON
            }

            promises.push(
                this.loadValueDataOnly(this.state.index, varType)
                    .then(valueData => {

                        // O JSON tem apenas um array de valores, indexado pela célula
                        if (valueData &&
                            Array.isArray(valueData.values) &&
                            foundCell.cellIndex >= 0 &&
                            foundCell.cellIndex < valueData.values.length) {

                            const loadedValue = valueData.values[foundCell.cellIndex];
                            allValues[varType] = {
                                value: loadedValue,
                                label: config.label,
                                unit: config.unit
                            };
                        } else {
                            // Se não conseguir carregar, marcar como ausente (sem valor padrão)
                            allValues[varType] = {
                                value: null, // Sem valor padrão
                                label: config.label,
                                unit: config.unit,
                                ausente: true // Marcador de dados ausentes
                            };


                        }
                    })
                    .catch((err) => {
                        // Captura erros de promise e marca como ausente
                        allValues[varType] = {
                            value: null, // Sem valor padrão
                            label: config.label,
                            unit: config.unit,
                            ausente: true // Marcador de dados ausentes
                        };
                    })
            );
        });

        return Promise.all(promises).then(() => allValues);
    }

    showSidebar() {
        const cell = this.state.selectedCell;
        const config = VARIABLES_CONFIG[this.state.type];
        const sidebar = document.getElementById('sidebar');
        const content = document.getElementById('sidebarContent');

        let html = `
            <div class="info-section">
                <div class="info-section-title">
                    <i class="fas fa-map-pin"></i> Localização
                </div>
                <div class="info-item">
                    <span class="info-label">Latitude</span>
                    <span class="info-value">${cell.lat.toFixed(4)}°</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Longitude</span>
                    <span class="info-value">${cell.lng.toFixed(4)}°</span>
                </div>
            </div>

            <div class="info-section">
                <div class="info-section-title">
                    <i class="fas fa-chart-line"></i> ${config.label}
                </div>
                <div class="info-item">
                    <span class="info-label">Valor</span>
                    <span class="info-value">${cell.value.toFixed(2)}<span class="info-unit">${config.unit}</span></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Data/Hora</span>
                    <span class="info-value">${this.calculateDateTimeFromIndex(this.state.index)}</span>
                </div>
            </div>
        `;

        const specificInfo = config.specificInfo(cell.value, cell.allValues);
        // Passar apenas os valores das variáveis como argumento também
        // Se specificInfo não conseguir usar allValues, usar valor único como fallback
        if (specificInfo) {
            html += `
                <div class="info-section variable-specific">
                    <div class="info-section-title">
                        <i class="fas fa-bolt"></i> ${specificInfo.title}
                    </div>
            `;

            specificInfo.items.forEach(item => {
                html += `
                    <div class="stat-card">
                        <div style="color: #666; font-size: 0.85rem;">
                            <i class="fas ${item.icon}"></i> ${item.label}
                        </div>
                        <div class="stat-card-value">
                            ${item.value}
                            <span class="stat-card-unit">${item.unit || ''}</span>
                        </div>
                    </div>
                `;
            });

            // Adicionar editor de parâmetros para variáveis de geração energética
            html += this.createParametersEditor(this.state.type);

            html += `</div>`;
        }

        // Adicionar aviso de carregamento de gráficos
        html += `
            <div class="info-section" id="chartsLoadingAlert" style="display: none; background: #e3f2fd; border-left: 4px solid #2196F3; padding: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; color: #1565c0; font-size: 0.9rem;">
                    <span class="loading-spinner" style="width: 16px; height: 16px; border-width: 2px;"></span>
                    <strong>Gerando gráficos...</strong>
                </div>
            </div>
        `;

        content.innerHTML = html;
        sidebar.classList.add('active');

        // Setup listeners para o editor de parâmetros
        this.setupParametersEditorListeners(this.state.type);
    }

    closeSidebar() {
        document.getElementById('sidebar').classList.remove('active');

        // Remover marcador visual ao fechar sidebar
        if (this.selectedMarker) {
            this.map.removeLayer(this.selectedMarker);
            this.selectedMarker = null;
        }

        // Limpar célula selecionada
        this.state.selectedCell = null;
    }

    toggleWindLayer(isEnabled) {
        if (this.state.type !== 'eolico') {
            console.warn('Camada de vento só disponível para eólico');
            return;
        }

        if (isEnabled) {
            this.renderWindVectors();
        } else {
            this.clearWindVectors();
        }
    }

    renderWindVectors() {
        const canvas = document.getElementById('windVectorCanvas');
        if (!canvas || !this.currentValueData) {
            console.warn('Canvas ou dados de vento não disponíveis');
            return;
        }

        const windData = this.currentValueData?.metadata?.wind;
        if (!windData || !windData.downsampled_linear_indices || windData.downsampled_linear_indices.length === 0) {
            console.warn('Dados de vento não disponíveis na metadata');
            return;
        }

        // Configurar canvas
        canvas.width = this.map.getSize().x;
        canvas.height = this.map.getSize().y;
        const ctx = canvas.getContext('2d');

        // Limpar canvas completamente
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Obter as layers do mapa
        if (!this.currentGeoJsonLayer) {
            console.warn('Camada GeoJSON não disponível');
            return;
        }

        const layers = this.currentGeoJsonLayer.getLayers();
        if (layers.length === 0) {
            console.warn('Nenhuma célula no mapa');
            return;
        }

        const linearIndices = windData.downsampled_linear_indices;
        const angles = windData.downsampled_angles;
        const magnitudes = windData.downsampled_magnitudes;

        if (angles.length === 0 || magnitudes.length === 0) {
            console.warn('Dados de vento vazios');
            return;
        }

        // Encontrar escala de magnitude para cores
        const minMag = Math.min(...magnitudes);
        const maxMag = Math.max(...magnitudes);
        const magRange = maxMag - minMag || 1;

        console.log(`Renderizando ${linearIndices.length} vetores de vento. Range: [${minMag.toFixed(2)}, ${maxMag.toFixed(2)}]`);

        // Quando há clipping (ES), é necessário mapear índices lineares para as camadas filtradas
        // Criar mapa de índices lineares originais para as camadas visíveis
        const layerIndexMap = new Map();

        layers.forEach((layer, visibleIdx) => {
            // Extrair o índice linear original do GeoJSON da camada
            if (layer.feature && layer.feature.properties && 'linear_index' in layer.feature.properties) {
                const originalLinearIdx = layer.feature.properties.linear_index;
                layerIndexMap.set(originalLinearIdx, visibleIdx);
            }
        });

        console.log(`Mapa criado: ${layerIndexMap.size} mapeamentos de índices disponíveis`);

        // Renderizar vetores - mapear linearIndices originais para as camadas visíveis
        linearIndices.forEach((originalLayerIdx, idx) => {
            try {
                // Se houver clipping, usar o mapa de índices; caso contrário, usar diretamente
                let actualLayerIdx;

                if (this.state.isClippedToEs && layerIndexMap.size > 0) {
                    // Com clipping: procurar no mapa
                    actualLayerIdx = layerIndexMap.get(originalLayerIdx);
                    if (actualLayerIdx === undefined) {
                        // Índice não encontrado no clipping (célula fora do ES)
                        return;
                    }
                } else {
                    // Sem clipping: usar diretamente
                    actualLayerIdx = originalLayerIdx;
                }

                // Verificar se o índice está dentro da faixa válida
                if (actualLayerIdx < 0 || actualLayerIdx >= layers.length) {
                    console.debug(`Índice de layer ${actualLayerIdx} fora do intervalo [0, ${layers.length - 1}]`);
                    return;
                }

                const targetLayer = layers[actualLayerIdx];
                if (!targetLayer || !targetLayer.getBounds) {
                    console.debug(`Layer ${actualLayerIdx} não é válida`);
                    return;
                }

                const bounds = targetLayer.getBounds();
                const center = bounds.getCenter();
                const point = this.map.latLngToContainerPoint(center);

                const angle = angles[idx];
                const magnitude = magnitudes[idx];

                // Verificar se o ponto está dentro do canvas
                if (point.x >= 0 && point.x <= canvas.width &&
                    point.y >= 0 && point.y <= canvas.height) {
                    this.drawWindArrow(ctx, point.x, point.y, angle, magnitude, minMag, maxMag, magRange);
                }
            } catch (e) {
                // Ignorar erros individuais de renderização silenciosamente
                console.debug('Erro renderizando vetor:', e.message);
            }
        });
    }

    drawWindArrow(ctx, x, y, angle, magnitude, minMag, maxMag, magRange) {
        // Tamanho da seta proporcional à magnitude
        const normalizedMag = (magnitude - minMag) / magRange;
        const arrowLength = 8 + normalizedMag * 16; // Min 8, max 24
        const lineWidth = 0.8 + normalizedMag * 1.2; // Min 0.8, max 2.0 (mais fino)
        const arrowHeadSize = 3 + normalizedMag * 2; // Min 3, max 5 (mais fino)

        // Converter ângulo meteorológico para ângulo Canvas (0 = up/norte, aumenta clockwise)
        const rad = (angle - 90) * Math.PI / 180;

        // Todas as setas em preto
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.7;

        const endX = x + arrowLength * Math.cos(rad);
        const endY = y + arrowLength * Math.sin(rad);

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Desenhar cabeça da seta
        const angle1 = rad + Math.PI / 6;
        const angle2 = rad - Math.PI / 6;

        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - arrowHeadSize * Math.cos(angle1), endY - arrowHeadSize * Math.sin(angle1));
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - arrowHeadSize * Math.cos(angle2), endY - arrowHeadSize * Math.sin(angle2));
        ctx.stroke();

        ctx.globalAlpha = 1.0;
    }

    clearWindVectors() {
        const canvas = document.getElementById('windVectorCanvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    showErrorMessage(message) {
        // Mostrar notificação toast com a mensagem de erro
        const alertDiv = document.createElement('div');
        alertDiv.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            background: #ff6b6b;
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 2000;
            font-weight: 500;
            max-width: 300px;
            animation: slideInRight 0.3s ease;
        `;
        alertDiv.textContent = message;

        document.body.appendChild(alertDiv);

        // Auto-remover após 3 segundos
        setTimeout(() => {
            alertDiv.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => alertDiv.remove(), 300);
        }, 3000);
    }

    updateSelectedCellData() {
        /**
         * Atualiza os dados da célula selecionada para um novo horário ou variável
         * Mantém o marcador e a posição, apenas atualiza as informações
         */
        if (!this.state.selectedCell) return;

        const cell = this.state.selectedCell;
        const config = VARIABLES_CONFIG[this.state.type];

        // Obter o valor da célula para a variável/horário atual
        if (this.currentValueData && Array.isArray(this.currentValueData.values)) {
            const cellIndex = cell.cellIndex;
            if (cellIndex >= 0 && cellIndex < this.currentValueData.values.length) {
                const newValue = this.currentValueData.values[cellIndex];

                // Atualizar valor da célula
                cell.value = newValue;

                // Recarregar TODOS os valores de TODAS as variáveis para o novo horário
                // Importante: loadAllVariableValuesForCell vai usar currentValueData já carregado e
                // fazer fetch dos outros JSONs para preencher allValues completamente
                this.loadAllVariableValuesForCell(cell).then(allValues => {
                    cell.allValues = allValues;
                    this.showSidebar();

                    // Marcador já é recriado em showGeoJsonLayer() quando o layer é carregado
                    // Não precisa recriar aqui
                }).catch(err => {
                    console.error('Erro ao atualizar dados da célula:', err);
                    this.showErrorMessage('Sem informações meteorológicas para este horário');
                    this.closeSidebar();
                });
            } else {
                this.showErrorMessage('Sem informações meteorológicas para este horário');
                this.closeSidebar();
            }
        } else {
            this.showErrorMessage('Sem informações meteorológicas disponíveis');
            this.closeSidebar();
        }
    }
}