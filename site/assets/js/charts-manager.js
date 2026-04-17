/**
 * CHARTS_MANAGER.js
 *
 * Gerenciador de gráficos temporais para o mapa interativo.
 *
 * OTIMIZAÇÕES:
 *   - loadTimeSeriesData: usa AbortController para cancelar requests obsoletos
 *   - Modal persistente: reutiliza instâncias do Chart.js (.update()) sem destroy/recreate
 *   - Sidebar livre de gráficos, mantendo apenas informações textuais rápidas
 */

class ChartsManager {
  constructor(app) {
    this.app = app;
    this.charts = new Map();
    this.timeSeriesData = {};
    this.abortController = null;

    this._setupModalListeners();
  }

  _setupModalListeners() {
    const closeBtn = document.getElementById("timeSeriesCloseBtn");
    const exportBtn = document.getElementById("timeSeriesExportBtn");
    const modal = document.getElementById("timeSeriesModal");

    if (closeBtn) closeBtn.addEventListener("click", () => this.closeModal());
    if (exportBtn) exportBtn.addEventListener("click", () => this.exportCurrentData());
    if (modal)
      modal.addEventListener("click", (e) => {
        if (e.target === modal) this.closeModal();
      });
  }

  // ─── Carregamento de Dados ────────────────────────────────────────────────

  async loadTimeSeriesData(lat, lng, domain) {
    // Abortar request anterior se houver
    if (this.abortController) {
      this.abortController.abort();
    }
    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    try {
      const cellIndex = await this.findCellIndex(lat, lng, domain, signal);
      if (cellIndex === null) return {};

      const timeSeriesData = {};
      const variableKeys = Object.keys(VARIABLES_CONFIG);

      // Processar variáveis em paralelo
      await Promise.all(
        variableKeys.map(async (variableKey) => {
          const config = VARIABLES_CONFIG[variableKey];
          if (!config?.id) return;

          let variableId = config.id;
          if (variableKey === "eolico" && this.app.windHeight) {
            if (this.app.windHeight === 100) variableId = config.id_100m;
            if (this.app.windHeight === 150) variableId = config.id_150m;
          }

          const BATCH_SIZE = 10;
          const allResults = [];
          for (let start = 0; start < 73; start += BATCH_SIZE) {
            if (signal.aborted) throw new DOMException("Aborted", "AbortError");
            const batch = Array.from({ length: Math.min(BATCH_SIZE, 73 - start) }, (_, j) => {
              const hour = start + j + 1;
              return this._fetchHourJson(variableId, domain, hour, signal)
                .then((data) => {
                  if (data?.values && Array.isArray(data.values)) {
                    const cellValue = data.values[cellIndex];
                    if (cellValue != null) {
                      return {
                        hour,
                        value: cellValue,
                        timestamp: this._timestampForHour(hour, data),
                      };
                    }
                  }
                  return null;
                })
                .catch((err) => {
                  if (err.name === "AbortError") throw err;
                  return null;
                });
            });
            const batchResults = await Promise.all(batch);
            allResults.push(...batchResults);
          }

          const hourlyData = allResults.filter(Boolean);

          if (hourlyData.length > 0) {
            timeSeriesData[variableKey] = { config, data: hourlyData };
          }
        })
      );

      if (signal.aborted) return {};

      this.timeSeriesData = timeSeriesData;
      return timeSeriesData;
    } catch (error) {
      if (error.name === "AbortError") {
        console.log("[Charts] Request cancelado pelo usuário.");
      } else {
        console.error("[Charts] Erro ao carregar série temporal:", error);
      }
      return {};
    }
  }

  async findCellIndex(lat, lng, domain, signal) {
    try {
      const cacheKey = domain;
      const cachedLayer = this.app?.gridLayers?.[cacheKey];

      if (cachedLayer) {
        const layers = cachedLayer.getLayers();
        let closestIndex = 0,
          minDist = Infinity;
        layers.forEach((layer, i) => {
          const bounds = layer.getBounds?.();
          if (!bounds) return;
          const c = bounds.getCenter();
          const d = this._quickDist(lat, lng, c.lat, c.lng);
          if (d < minDist) {
            minDist = d;
            closestIndex = i;
          }
        });
        return closestIndex;
      }

      const res = await fetch(`GeoJSON/${domain}.geojson`, { signal });
      if (!res.ok) return null;
      const geoJson = await res.json();

      let closestIndex = 0,
        minDist = Infinity;
      (geoJson.features || []).forEach((feature, i) => {
        if (feature.geometry?.type === "Polygon") {
          const c = this._centroid(feature.geometry.coordinates[0]);
          const d = this._quickDist(lat, lng, c.lat, c.lng);
          if (d < minDist) {
            minDist = d;
            closestIndex = i;
          }
        }
      });
      return closestIndex;
    } catch (error) {
      if (error.name !== "AbortError") console.error("[Charts] Erro ao encontrar célula:", error);
      throw error;
    }
  }

  // ─── Renderização Modal ───────────────────────────────────────────────────

  openModal() {
    const modal = document.getElementById("timeSeriesModal");
    if (modal) modal.style.display = "flex";
  }

  closeModal() {
    const modal = document.getElementById("timeSeriesModal");
    if (modal) modal.style.display = "none";
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  renderChartsForVariable(variableType, selectedCellData) {
    const config = VARIABLES_CONFIG[variableType];
    if (!config) return;

    document.getElementById("timeSeriesModalTitle").innerHTML =
      `<i class="fas fa-${this._getIcon(variableType, "value")}"></i> Série Temporal: ${config.label}`;

    const isSolarOrWind = variableType === "solar" || variableType === "eolico";

    this._updateOrCreateChart(variableType, "value", "chartCanvasValue");

    const energyContainer = document.getElementById("chartContainerEnergy");
    if (isSolarOrWind) {
      energyContainer.style.display = "block";
      this._updateOrCreateChart(variableType, "energy", "chartCanvasEnergy");
    } else {
      energyContainer.style.display = "none";
    }
  }

  reloadChartsWithNewParameters() {
    const { type, selectedCell } = this.app?.state || {};
    if (type && selectedCell && this.timeSeriesData) {
      this.renderChartsForVariable(type, selectedCell);
    }
  }

  clearCharts() {
    this.charts.forEach((chart) => chart?.destroy());
    this.charts.clear();
    this.timeSeriesData = {};
  }

  // ─── Internos ────────────────────────────────────────────────────────────

  _updateOrCreateChart(variableType, chartType, canvasId) {
    if (!this.timeSeriesData?.[variableType]) return;

    const config = VARIABLES_CONFIG[variableType];
    const timeData = this.timeSeriesData[variableType].data;
    const {
      data: chartData,
      label: chartLabel,
      unit: chartUnit,
      color: chartColor,
    } = this._prepareChartData(variableType, chartType, config, timeData);

    // Otimização: Cache das datas parseadas
    const labels = timeData.map((d) => {
      if (!d._formattedTime) {
        d._formattedTime = new Date(d.timestamp).toLocaleTimeString("pt-BR", {
          hour: "2-digit",
          minute: "2-digit",
        });
      }
      return d._formattedTime;
    });

    let chartInstance = this.charts.get(canvasId);

    if (chartInstance) {
      // Atualizar in-place (Performance Win)
      chartInstance.data.labels = labels;
      chartInstance.data.datasets[0].data = chartData;
      chartInstance.data.datasets[0].label = chartLabel;
      chartInstance.data.datasets[0].borderColor = chartColor;
      chartInstance.data.datasets[0].backgroundColor = `${chartColor}20`;
      chartInstance.data.datasets[0].pointBackgroundColor = chartColor;
      chartInstance.options.scales.y.title.text = chartUnit;
      chartInstance.options.plugins.tooltip.callbacks.label = (ctx) => `${ctx.parsed.y.toFixed(2)} ${chartUnit}`;
      chartInstance.update();
    } else {
      // Criar novo se não existe
      const ctx = document.getElementById(canvasId).getContext("2d");
      chartInstance = new Chart(ctx, this._buildChartConfig(chartData, labels, chartLabel, chartColor, chartUnit));
      this.charts.set(canvasId, chartInstance);
    }
  }

  _buildChartConfig(chartData, labels, chartLabel, chartColor, chartUnit) {
    return {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: chartLabel,
            data: chartData,
            borderColor: chartColor,
            backgroundColor: `${chartColor}20`,
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: chartColor,
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              font: { size: 14 },
              color: "#666",
              padding: 15,
              usePointStyle: true,
            },
          },
          tooltip: {
            backgroundColor: "rgba(0,0,0,0.8)",
            titleColor: "#fff",
            bodyColor: "#fff",
            borderColor: chartColor,
            borderWidth: 2,
            padding: 12,
            displayColors: false,
            callbacks: {
              label: (ctx) => `${ctx.parsed.y.toFixed(2)} ${chartUnit}`,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: {
              color: "#888",
              font: { size: 13 },
              callback: (v) => v.toFixed(1),
            },
            grid: { color: "#f0f0f0", drawBorder: false },
            title: {
              display: true,
              text: chartUnit,
              font: { size: 13, weight: "bold" },
              color: "#666",
            },
          },
          x: {
            ticks: { color: "#888", font: { size: 13 } },
            grid: { color: "#f0f0f0", drawBorder: false },
          },
        },
      },
    };
  }

  _prepareChartData(variableType, chartType, config, timeData) {
    if (chartType === "value") {
      return {
        data: timeData.map((d) => d.value),
        label: config.label,
        unit: config.unit,
        color: config.colors[config.colors.length - 1],
      };
    }

    const unit = variableType === "solar" ? "Wh/m²" : "kWh";
    const color = variableType === "solar" ? "#FDB462" : "#80B1D3";
    const data = timeData.map((d) => {
      try {
        const info = config.specificInfo(d.value, {});
        const item = info?.items?.find(
          (it) => it.label?.includes("Produção Energética") || it.label?.includes("kWh") || it.label?.includes("Wh")
        );
        if (item?.value) {
          const num = parseFloat(
            String(item.value)
              .replace(/[^\d.,]/g, "")
              .replace(",", ".")
          );
          return isNaN(num) ? 0 : num;
        }
      } catch (_) {}
      return 0;
    });

    return { data, label: "Produção Energética Acumulada (1h)", unit, color };
  }

  exportCurrentData() {
    const { type, selectedCell } = this.app?.state || {};
    if (!type || !selectedCell || !this.timeSeriesData?.[type]) return;

    const config = VARIABLES_CONFIG[type];
    const timeData = this.timeSeriesData[type].data;
    const chartDataValue = this._prepareChartData(type, "value", config, timeData).data;

    let csv = `Data,Hora,Latitude,Longitude,Variável,Valor(${config.unit})`;
    const isEnergy = type === "solar" || type === "eolico";
    let chartDataEnergy = null;
    let energyUnit = "";

    if (isEnergy) {
      const energyConfig = this._prepareChartData(type, "energy", config, timeData);
      chartDataEnergy = energyConfig.data;
      energyUnit = energyConfig.unit;
      csv += `,Produção(${energyUnit})`;
    }
    csv += "\n";

    timeData.forEach((d, i) => {
      const date = new Date(d.timestamp);
      const dateStr = date.toLocaleDateString("pt-BR");
      const timeStr = date.toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      const rawV = chartDataValue[i];
      const numV =
        typeof rawV === "string" ? parseFloat(rawV.replace(/[^\d.,]/g, "").replace(",", ".")) : parseFloat(rawV);

      csv += `${dateStr},${timeStr},${selectedCell.lat.toFixed(4)},${selectedCell.lng.toFixed(4)},"${config.label}",${isNaN(numV) ? "0.00" : numV.toFixed(2)}`;

      if (isEnergy && chartDataEnergy) {
        const rawE = chartDataEnergy[i];
        const numE =
          typeof rawE === "string" ? parseFloat(rawE.replace(/[^\d.,]/g, "").replace(",", ".")) : parseFloat(rawE);
        csv += `,${isNaN(numE) ? "0.00" : numE.toFixed(2)}`;
      }
      csv += "\n";
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `timeseries_${type}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async _fetchHourJson(variableId, domain, hour, signal) {
    const id_num = String(hour).padStart(3, "0");
    const url = `JSON/${domain}_${variableId}_${id_num}.json`;
    try {
      if (this.app?._cachedFetch) {
        return await this.app._cachedFetch(url, { signal });
      }
      const res = await fetch(url, { signal });
      return res.ok ? res.json() : null;
    } catch (e) {
      if (e.name === "AbortError") throw e;
      return null;
    }
  }

  _timestampForHour(hour, data) {
    const meta = data?.metadata;
    if (meta?.start_date) {
      const base = new Date(meta.start_date);
      if (!isNaN(base)) {
        base.setHours(base.getHours() + (hour - 1));
        return base.toISOString();
      }
    }
    const base = new Date();
    base.setMinutes(0, 0, 0);
    base.setHours(base.getHours() + (hour - 1));
    return base.toISOString();
  }

  _quickDist(lat1, lng1, lat2, lng2) {
    const dlat = lat1 - lat2;
    const dlng = lng1 - lng2;
    return dlat * dlat + dlng * dlng;
  }

  _centroid(coords) {
    let lat = 0,
      lng = 0;
    const n = coords.length - 1;
    for (let i = 0; i < n; i++) {
      lng += coords[i][0];
      lat += coords[i][1];
    }
    return { lat: lat / n, lng: lng / n };
  }

  _getIcon(variableType, chartType) {
    if (chartType === "energy") return variableType === "solar" ? "solar-panel" : "fan";
    return (
      {
        solar: "sun",
        eolico: "wind",
        temperature: "thermometer",
        pressure: "cloud",
        humidity: "droplet",
        rain: "cloud-rain",
      }[variableType] || "chart-line"
    );
  }
}
