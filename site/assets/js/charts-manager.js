/**
 * CHARTS_MANAGER.js
 *
 * Gerenciador de gráficos temporais para o mapa interativo.
 * Responsável por renderizar gráficos de evolução temporal de variáveis
 * e exportação de dados em CSV.
 *
 * OTIMIZAÇÕES:
 *   - loadTimeSeriesData: fetches das 73 horas paralelizados com Promise.all
 *     (antes: sequencial — até 438 requests seriais; agora: paralelo por variável)
 *   - findCellIndex: reutiliza gridLayers já carregados no app antes de fazer fetch
 *   - buildChartConfig: config compartilhada entre sidebar e modal (sem duplicação)
 *   - Console.logs de debug removidos
 */

class ChartsManager {
    constructor(app) {
        this.app = app;
        this.charts = new Map();
        this.timeSeriesData = {};
    }

    // ─── Carregamento de Dados ────────────────────────────────────────────────

    /**
     * Carrega dados temporais de uma célula selecionada.
     * Fetches de TODAS as horas de uma variável são feitos em PARALELO
     * via Promise.all, reduzindo o tempo de carregamento drasticamente.
     */
    async loadTimeSeriesData(lat, lng, domain) {
        try {
            const cellIndex = await this.findCellIndex(lat, lng, domain);
            if (cellIndex === null) return {};

            const timeSeriesData = {};
            const variableKeys = Object.keys(VARIABLES_CONFIG);

            // Processar todas as variáveis em paralelo
            await Promise.all(variableKeys.map(async (variableKey) => {
                const config = VARIABLES_CONFIG[variableKey];
                if (!config?.id) return;

                let variableId = config.id;
                if (variableKey === 'eolico' && this.app.windHeight) {
                    if (this.app.windHeight === 100) variableId = config.id_100m;
                    if (this.app.windHeight === 150) variableId = config.id_150m;
                }

                // Fetch hours in batches of 10 to avoid network saturation
                const BATCH_SIZE = 10;
                const allResults = [];
                for (let start = 0; start < 73; start += BATCH_SIZE) {
                    const batch = Array.from(
                        { length: Math.min(BATCH_SIZE, 73 - start) },
                        (_, j) => {
                            const hour = start + j + 1;
                            return this._fetchHourJson(variableId, domain, hour)
                                .then(data => {
                                    if (data?.values && Array.isArray(data.values)) {
                                        const cellValue = data.values[cellIndex];
                                        if (cellValue != null) {
                                            return { hour, value: cellValue, timestamp: this._timestampForHour(hour, data) };
                                        }
                                    }
                                    return null;
                                })
                                .catch(() => null);
                        }
                    );
                    const batchResults = await Promise.all(batch);
                    allResults.push(...batchResults);
                }

                const hourlyData = allResults.filter(Boolean);

                if (hourlyData.length > 0) {
                    timeSeriesData[variableKey] = { config, data: hourlyData };
                }
            }));

            this.timeSeriesData = timeSeriesData;
            return timeSeriesData;
        } catch (error) {
            console.error('[Charts] Erro ao carregar série temporal:', error);
            return {};
        }
    }

    /**
     * Encontra o índice da célula mais próxima das coordenadas fornecidas.
     * Reutiliza o gridLayer já carregado no app sempre que possível,
     * evitando um fetch duplicado do GeoJSON.
     */
    async findCellIndex(lat, lng, domain) {
        try {
            // Use domain-only cache key (geometry is shared across all variables)
            const cacheKey = domain;
            const cachedLayer = this.app?.gridLayers?.[cacheKey];

            if (cachedLayer) {
                const layers = cachedLayer.getLayers();
                let closestIndex = 0, minDist = Infinity;
                layers.forEach((layer, i) => {
                    const bounds = layer.getBounds?.();
                    if (!bounds) return;
                    const c = bounds.getCenter();
                    const d = this._quickDist(lat, lng, c.lat, c.lng);
                    if (d < minDist) { minDist = d; closestIndex = i; }
                });
                return closestIndex;
            }

            // Fallback: fetch domain-only GeoJSON
            const res = await fetch(`geoJSON/${domain}.geojson`);
            if (!res.ok) return null;
            const geoJson = await res.json();

            let closestIndex = 0, minDist = Infinity;
            (geoJson.features || []).forEach((feature, i) => {
                if (feature.geometry?.type === 'Polygon') {
                    const c = this._centroid(feature.geometry.coordinates[0]);
                    const d = this._quickDist(lat, lng, c.lat, c.lng);
                    if (d < minDist) { minDist = d; closestIndex = i; }
                }
            });
            return closestIndex;
        } catch (error) {
            console.error('[Charts] Erro ao encontrar índice da célula:', error);
            return null;
        }
    }

    // ─── Renderização ─────────────────────────────────────────────────────────

    /** Renderiza gráficos no sidebar para a variável atual. */
    renderChartsForVariable(variableType, selectedCellData) {
        const sidebarContent = document.getElementById('sidebarContent');
        if (!sidebarContent) return;

        sidebarContent.querySelectorAll('.chart-container').forEach(el => el.remove());

        const isSolarOrWind = variableType === 'solar' || variableType === 'eolico';
        this._renderChart(variableType, 'value', selectedCellData, sidebarContent);
        if (isSolarOrWind) {
            this._renderChart(variableType, 'energy', selectedCellData, sidebarContent);
        }
    }

    /** Recarrega gráficos com novos parâmetros (ex: após slider de parâmetros). */
    reloadChartsWithNewParameters() {
        const { type, selectedCell } = this.app?.state || {};
        if (type && selectedCell && this.timeSeriesData) {
            this.renderChartsForVariable(type, selectedCell);
        }
    }

    /** Destrói todos os gráficos e limpa o cache. */
    clearCharts() {
        this.charts.forEach(chart => chart?.destroy());
        this.charts.clear();
        this.timeSeriesData = {};
    }

    // ─── Internos ────────────────────────────────────────────────────────────

    /**
     * Renderiza um gráfico individual (sidebar ou modal compartilham esta lógica).
     */
    _renderChart(variableType, chartType, selectedCellData, container) {
        if (!this.timeSeriesData?.[variableType]) return;

        const config   = VARIABLES_CONFIG[variableType];
        const timeData = this.timeSeriesData[variableType].data;
        const { data: chartData, label: chartLabel, unit: chartUnit, color: chartColor } =
            this._prepareChartData(variableType, chartType, config, timeData);

        // Montar container HTML
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <div class="chart-header">
                <div class="chart-title">
                    <i class="fas fa-${this._getIcon(variableType, chartType)}"></i>
                    ${chartLabel}
                </div>
                <div class="chart-buttons">
                    <button class="chart-expand-btn" title="Expandir gráfico">
                        <i class="fas fa-expand"></i>
                    </button>
                    <button class="chart-export-btn">
                        <i class="fas fa-download"></i> CSV
                    </button>
                </div>
            </div>
            <div class="chart-canvas-wrapper">
                <canvas id="chart-${variableType}-${chartType}"></canvas>
            </div>
        `;
        container.appendChild(chartContainer);

        // Destruir instância anterior se existir
        const chartKey = `${variableType}-${chartType}`;
        this.charts.get(chartKey)?.destroy();

        const ctx = document.getElementById(`chart-${variableType}-${chartType}`).getContext('2d');
        const labels = timeData.map(d => {
            const date = new Date(d.timestamp);
            return date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
        });

        const instance = new Chart(ctx, this._buildChartConfig(chartData, labels, chartLabel, chartColor, chartUnit, false));
        this.charts.set(chartKey, instance);

        // Listeners
        chartContainer.querySelector('.chart-export-btn').addEventListener('click', () => {
            this._exportCSV(variableType, chartType, selectedCellData, timeData, chartData, config, chartUnit);
        });
        chartContainer.querySelector('.chart-expand-btn').addEventListener('click', () => {
            this._expandChart(variableType, chartType, chartLabel, timeData, chartData, chartColor, chartUnit);
        });
    }

    /** Abre gráfico em modal em tela cheia. */
    _expandChart(variableType, chartType, chartLabel, timeData, chartData, chartColor, chartUnit) {
        const modal = document.createElement('div');
        modal.className = 'chart-modal-overlay';
        modal.innerHTML = `
            <div class="chart-modal-content">
                <div class="chart-modal-header">
                    <h2>${chartLabel}</h2>
                    <button class="chart-modal-close" title="Fechar">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="chart-modal-body">
                    <canvas id="chart-expanded-${variableType}-${chartType}"></canvas>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const labels = timeData.map(d => {
            const date = new Date(d.timestamp);
            return date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
        });
        const ctx = document.getElementById(`chart-expanded-${variableType}-${chartType}`).getContext('2d');
        const expandedChart = new Chart(ctx, this._buildChartConfig(chartData, labels, chartLabel, chartColor, chartUnit, true));

        const close = () => { expandedChart.destroy(); modal.remove(); };
        modal.querySelector('.chart-modal-close').addEventListener('click', close);
        modal.addEventListener('click', e => { if (e.target === modal) close(); });
    }

    /**
     * Constrói a configuração do Chart.js.
     * Compartilhada entre sidebar e modal — elimina duplicação de código.
     * @param {boolean} expanded — true para modal (fontes maiores)
     */
    _buildChartConfig(chartData, labels, chartLabel, chartColor, chartUnit, expanded) {
        const fs = expanded ? { tick: 13, legend: 14, title: 13 } : { tick: 11, legend: 12, title: 11 };
        return {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: chartLabel,
                    data: chartData,
                    borderColor: chartColor,
                    backgroundColor: `${chartColor}20`,
                    borderWidth: expanded ? 3 : 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: expanded ? 5 : 4,
                    pointBackgroundColor: chartColor,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: expanded ? 7 : 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { font: { size: fs.legend }, color: '#666', padding: expanded ? 15 : 12, usePointStyle: true }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: chartColor,
                        borderWidth: expanded ? 2 : 1,
                        padding: expanded ? 12 : 10,
                        displayColors: false,
                        callbacks: { label: ctx => `${ctx.parsed.y.toFixed(2)} ${chartUnit}` }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: { color: '#888', font: { size: fs.tick }, callback: v => v.toFixed(1) },
                        grid: { color: '#f0f0f0', drawBorder: false },
                        title: {
                            display: true,
                            text: chartUnit,
                            ...(expanded ? { font: { size: fs.title, weight: 'bold' }, color: '#666' } : {})
                        }
                    },
                    x: {
                        ticks: { color: '#888', font: { size: fs.tick } },
                        grid: { color: '#f0f0f0', drawBorder: false }
                    }
                }
            }
        };
    }

    /**
     * Prepara os dados de um gráfico (values ou energy) para Chart.js.
     */
    _prepareChartData(variableType, chartType, config, timeData) {
        if (chartType === 'value') {
            return {
                data:  timeData.map(d => d.value),
                label: config.label,
                unit:  config.unit,
                color: config.colors[config.colors.length - 1]
            };
        }

        // chartType === 'energy'
        const unit  = variableType === 'solar' ? 'Wh/m²' : 'kWh';
        const color = variableType === 'solar' ? '#FDB462' : '#80B1D3';
        const data  = timeData.map(d => {
            try {
                const info = config.specificInfo(d.value, {});
                const item = info?.items?.find(it =>
                    it.label?.includes('Produção Energética') ||
                    it.label?.includes('kWh') ||
                    it.label?.includes('Wh')
                );
                if (item?.value) {
                    const num = parseFloat(String(item.value).replace(/[^\d.,]/g, '').replace(',', '.'));
                    return isNaN(num) ? 0 : num;
                }
            } catch (_) {}
            return 0;
        });

        return { data, label: 'Produção Energética Acumulada (1h)', unit, color };
    }

    /** Exporta os dados do gráfico para CSV. */
    _exportCSV(variableType, chartType, cellData, timeData, chartData, config, chartUnit) {
        const isEnergy = chartType === 'energy';
        const varName  = isEnergy
            ? (variableType === 'solar' ? 'Geração Solar' : 'Geração Eólica')
            : config.label;
        const unit = isEnergy
            ? (variableType === 'solar' ? 'Wh/m²' : 'kWh')
            : config.unit;

        let csv = `Data,Hora,Latitude,Longitude,Variável,Valor(${unit})\n`;

        timeData.forEach((d, i) => {
            const date    = new Date(d.timestamp);
            const dateStr = date.toLocaleDateString('pt-BR');
            const timeStr = date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const raw     = chartData[i];
            const num     = typeof raw === 'string'
                ? parseFloat(raw.replace(/[^\d.,]/g, '').replace(',', '.'))
                : parseFloat(raw);
            csv += `${dateStr},${timeStr},${cellData.lat.toFixed(4)},${cellData.lng.toFixed(4)},"${varName}",${isNaN(num) ? '0.00' : num.toFixed(2)}\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url  = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const ts   = new Date().toISOString().slice(0, 10);
        link.href     = url;
        link.download = `timeseries_${variableType}${isEnergy ? '_energy' : ''}_${ts}.csv`;
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    // ─── Utilitários ─────────────────────────────────────────────────────────

    /** Fetch de um JSON de hora específica — usa cache compartilhado do app. */
    async _fetchHourJson(variableId, domain, hour) {
        const id_num = String(hour).padStart(3, '0');
        const url = `JSON/${domain}_${variableId}_${id_num}.json`;
        try {
            // Use app's cached fetch if available (benefits from Web Worker + memory cache)
            if (this.app?._cachedFetch) {
                return await this.app._cachedFetch(url);
            }
            const res = await fetch(url);
            return res.ok ? res.json() : null;
        } catch (_) {
            return null;
        }
    }

    /**
     * Calcula o timestamp para uma hora usando a data inicial do WRF do app,
     * com fallback para new Date() se não disponível.
     */
    _timestampForHour(hour, data) {
        // Tentar extrair data do metadata do JSON
        const meta = data?.metadata;
        if (meta?.start_date) {
            const base = new Date(meta.start_date);
            if (!isNaN(base)) {
                base.setHours(base.getHours() + (hour - 1));
                return base.toISOString();
            }
        }
        // Fallback
        const base = new Date();
        base.setMinutes(0, 0, 0);
        base.setHours(base.getHours() + (hour - 1));
        return base.toISOString();
    }

    /** Distância aproximada (quadrática) entre dois pontos — suficiente para comparação relativa. */
    _quickDist(lat1, lng1, lat2, lng2) {
        const dlat = lat1 - lat2;
        const dlng = lng1 - lng2;
        return dlat * dlat + dlng * dlng;
    }

    /** Centróide simples de um polígono. */
    _centroid(coords) {
        let lat = 0, lng = 0;
        const n = coords.length - 1;
        for (let i = 0; i < n; i++) { lng += coords[i][0]; lat += coords[i][1]; }
        return { lat: lat / n, lng: lng / n };
    }

    /** Ícone FontAwesome para o tipo de gráfico. */
    _getIcon(variableType, chartType) {
        if (chartType === 'energy') return variableType === 'solar' ? 'solar-panel' : 'fan';
        return { solar: 'sun', eolico: 'wind', temperature: 'thermometer',
                 pressure: 'cloud', humidity: 'droplet', rain: 'cloud-rain' }[variableType] || 'chart-line';
    }
}
