/**
 * INICIALIZAÇÃO DA APLICAÇÃO
 */
let app;
let chartsManager;

document.addEventListener('DOMContentLoaded', () => {
    app = new MeteoMapManager();

    // Inicializar gerenciador de gráficos
    chartsManager = new ChartsManager(app);

    // Anexar chartsManager ao app para uso em handleMapClick
    app.chartsManager = chartsManager;

    // Interceptar o método showSidebar para renderizar gráficos também
    const originalShowSidebar = app.showSidebar.bind(app);
    app.showSidebar = function () {
        // Chamar o método original
        originalShowSidebar();

        // Renderizar gráficos após mostrar a sidebar
        if (app.state.selectedCell) {
            // Mostrar aviso de carregamento
            const loadingAlert = document.getElementById('chartsLoadingAlert');
            if (loadingAlert) {
                loadingAlert.style.display = 'block';
            }

            chartsManager.loadTimeSeriesData(
                app.state.selectedCell.lat,
                app.state.selectedCell.lng,
                app.getDomainFromZoom(app.map.getZoom())
            ).then(() => {
                chartsManager.renderChartsForVariable(
                    app.state.type,
                    app.state.selectedCell
                );

                // Esconder aviso de carregamento
                setTimeout(() => {
                    const loadingAlert = document.getElementById('chartsLoadingAlert');
                    if (loadingAlert) {
                        loadingAlert.style.display = 'none';
                    }
                }, 300);
            }).catch(err => {
                console.error('Erro ao carregar e renderizar gráficos:', err);

                // Esconder aviso em caso de erro também
                const loadingAlert = document.getElementById('chartsLoadingAlert');
                if (loadingAlert) {
                    loadingAlert.style.display = 'none';
                }
            });
        }
    };
});