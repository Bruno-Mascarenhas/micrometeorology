/**
 * INICIALIZAÇÃO DA APLICAÇÃO
 */
let app;
let chartsManager;

document.addEventListener("DOMContentLoaded", () => {
  app = new MeteoMapManager();

  // Inicializar gerenciador de gráficos
  chartsManager = new ChartsManager(app);

  // Anexar chartsManager ao app para uso em handleMapClick
  app.chartsManager = chartsManager;

  // Interceptar o método showSidebar para renderizar gráficos também
  const originalShowSidebar = app.showSidebar.bind(app);
  app.showSidebar = function () {
    originalShowSidebar();

    if (app.state.selectedCell) {
      // Open modal and show overlay
      chartsManager.openModal();
      const loadingOverlay = document.getElementById("timeSeriesLoadingOverlay");
      if (loadingOverlay) loadingOverlay.style.display = "flex";

      chartsManager
        .loadTimeSeriesData(
          app.state.selectedCell.lat,
          app.state.selectedCell.lng,
          app.getDomainFromZoom(app.map.getZoom())
        )
        .then((data) => {
          // If it returned empty object, it might have been aborted
          if (Object.keys(data).length > 0) {
            chartsManager.renderChartsForVariable(app.state.type, app.state.selectedCell);
          }
          if (loadingOverlay) loadingOverlay.style.display = "none";
        })
        .catch((err) => {
          console.error("Erro ao carregar e renderizar gráficos:", err);
          if (loadingOverlay) loadingOverlay.style.display = "none";
        });
    }
  };
});
