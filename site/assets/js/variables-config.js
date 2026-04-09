/**
 * VARIABLES_CONFIG.js
 * 
 * Configuração centralizada de variáveis meteorológicas.
 * Adicione novas variáveis aqui para expandir a funcionalidade do sistema.
 * 
 * Cada variável contém:
 * - id: Identificador do arquivo (ex: SWDOWN, POT_EOLICO_100M)
 * - label: Nome exibido na UI
 * - unit: Unidade de medida
 * - colormap: Tipo de paleta de cores
 * - colors: Array de cores hex em gradiente
 * - specificInfo: Função que retorna informações específicas da variável
 *   * Agora recebe: (value, allValues = {})
 *   * value: Valor da variável atual
 *   * allValues: Objeto com valores de TODAS as variáveis para a mesma célula/data
 *     Exemplo: { temperature: { value: 25, label: '...', unit: '°C' }, ... }
 *   * Permite cálculos multivariáveis (ex: produção solar com ajuste de temperatura)
 */

/**
 * Helper para obter parâmetros customizados ou usar padrão
 * @param {string} variableType - Tipo da variável (solar, eolico, etc)
 * @param {string} paramName - Nome do parâmetro
 * @param {number} defaultValue - Valor padrão se não customizado
 * @returns {number} Valor customizado ou padrão
 */
function getParameter(variableType, paramName, defaultValue) {
    if (typeof app === 'undefined' || !app || !app.getCustomParameter) {
        return defaultValue;
    }
    
    try {
        const customValue = app.getCustomParameter(variableType, paramName);
        if (customValue !== null && customValue !== undefined) {
            return customValue;
        }
    } catch (e) {
        console.warn(`Erro ao obter parâmetro: ${e.message}`);
    }
    
    return defaultValue;
}

const VARIABLES_CONFIG = {
    solar: {
        id: 'SWDOWN',
        label: 'Radiação Solar',
        unit: 'W/m²',
        colormap: 'hot_r',
        colors: ['#ffffff', '#fff0a0', '#ffd700', '#ffaa00', '#ff6600', '#ff2200', '#dd0000', '#aa0000', '#7a0000', '#691009'],
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente, retornar aviso
            if (value === null || value === undefined) {
                return {
                    title: 'Geração Fotovoltaica',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            const air_temp = allValues.temperature?.value || 25;
            const panelEfficiency = getParameter('solar', 'panelEfficiency', 18) / 100;
            const inversorEfficiency = getParameter('solar', 'inversorEfficiency', 95) / 100;
            const ptc = getParameter('solar', 'ptc', -0.38);
            const noct = getParameter('solar', 'noct', 45);
            
            const nominal_params_temp = 25;
            const foto_cell_temp = air_temp + ((noct - 20) * value / 800);
            const energy_gen = (value/1000) * panelEfficiency * inversorEfficiency * (1 + ptc * (foto_cell_temp - nominal_params_temp) / 100);
            
            return {
                title: 'Geração Fotovoltaica',
                items: [
                    {
                        label: 'Radiação Incidente Acumulada (1h)',
                        value: (value * 3.6).toFixed(2),
                        unit: 'kJ/m²',
                        icon: 'fa-sun'
                    },
                    { 
                        label: 'Produção Energética Acumulada (1h)', 
                        value: (energy_gen * 1000).toFixed(2), 
                        unit: 'Wh/m²',
                        icon: 'fa-solar-panel' 
                    },
                ]
            }
        }
    },

    eolico: {
        id: 'POT_EOLICO_50M',
        id_100m: 'POT_EOLICO_100M',
        id_150m: 'POT_EOLICO_150M',
        defaultHeight: 50,
        label: 'Velocidade do Vento',
        unit: 'm/s',
        colormap: 'Blues',
        colors: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente ou ausente flag ativo, retornar aviso
            if (value === null || value === undefined || allValues.eolico?.ausente) {
                return {
                    title: 'Geração Eólica',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            const tempValue = allValues.temperature?.value || 15;
            
            // Usar parâmetros customizados ou padrão
            const airDensity = getParameter('eolico', 'airDensity', 1.225);
            const rotorDiameter = getParameter('eolico', 'rotorDiameter', 40);
            const Cp = getParameter('eolico', 'powerCoefficient', 0.4);
            
            const densityOfAir = airDensity * (288 / (273 + tempValue)); // Densidade corrigida por temperatura
            const rotorArea = Math.PI * Math.pow(rotorDiameter / 2, 2);
            
            return {
                title: 'Geração Eólica',
                items: [
                    { 
                        label: 'Categoria do Vento',
                        value: getWindCategory(value),
                        icon: 'fa-wind'
                    },
                    { 
                        label: 'Densidade de Potência',
                        value: (0.5 * densityOfAir * Math.pow(value, 3)).toFixed(0),
                        unit: 'W/m²',
                        icon: 'fa-fan'
                    },
                    {
                        label: `Produção Energética Acumulada (1h)`,
                        value: (0.5 * densityOfAir * Math.pow(value, 3) * rotorArea * Cp / 1000).toFixed(1),
                        unit: 'kWh',
                        icon: 'fa-wind'
                    },
                ]
            }
        }
    },

    temperature: {
        id: 'TEMP',
        label: 'Temperatura (2m)',
        unit: '°C',
        colormap: 'hot_r',
        colors: ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000'],
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente ou ausente flag ativo, retornar aviso
            if (value === null || value === undefined || allValues.temperature?.ausente) {
                return {
                    title: 'Informações Térmicas',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            // Usar umidade se disponível para cálculo de sensação térmica
            const humidityValue = allValues.humidity?.value || 60;
            const windValue = allValues.wind?.value || 2;
            
            // Calcular sensação térmica considerando umidade
            const feelsLike = getTemperatureFeelsLike(value, humidityValue, windValue);
            const heatIndex = getHeatIndex(value, humidityValue);
            
            return {
                title: 'Informações Térmicas',
                items: [
                    { 
                        label: 'Sensação Térmica',
                        value: feelsLike.toFixed(1),
                        unit: '°C',
                        icon: 'fa-thermometer'
                    },
                    { 
                        label: 'Classificação',
                        value: value > 25 ? 'Quente' : (value < 15 ? 'Frio' : 'Moderado'),
                        icon: 'fa-info-circle'
                    },
                    {
                        label: 'Índice de Calor',
                        value: heatIndex.toFixed(1),
                        unit: '°C',
                        icon: 'fa-fire'
                    },
                ]
            }
        }
    },

    pressure: {
        id: 'PRES',
        label: 'Pressão Atmosférica',
        unit: 'hPa',
        colormap: 'RdBu_r',
        colors: ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'],
        useDynamicScale: true,
        normalValue: 1013,
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente ou ausente flag ativo, retornar aviso
            if (value === null || value === undefined || allValues.pressure?.ausente) {
                return {
                    title: 'Condições Atmosféricas',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            const tempValue = allValues.temperature?.value || 15;
            const humidityValue = allValues.humidity?.value || 60;
            
            // Calcular ponto de orvalho (aprox)
            const dewpoint = tempValue - (100 - humidityValue) / 5;
            
            return {
                title: 'Condições Atmosféricas',
                items: [
                    { 
                        label: 'Classificação',
                        value: value > 1013 ? 'Alta Pressão' : 'Baixa Pressão',
                        icon: 'fa-cloud'
                    },
                    { 
                        label: 'Tendência',
                        value: 'Estável',
                        icon: 'fa-chart-line'
                    },
                    {
                        label: 'Desvio Normal',
                        value: (value - 1013).toFixed(1),
                        unit: 'hPa',
                        icon: 'fa-arrow-up'
                    },
                ]
            }
        }
    },

    humidity: {
        id: 'VAPOR',
        label: 'Umidade Relativa',
        unit: '%',
        colormap: 'YlGnBu',
        colors: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fbc41', '#365f0f'],
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente ou ausente flag ativo, retornar aviso
            if (value === null || value === undefined || allValues.humidity?.ausente) {
                return {
                    title: 'Condições de Umidade',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            const tempValue = allValues.temperature?.value || 20;
            
            // Calcular ponto de orvalho
            const dewpoint = tempValue - (100 - value) / 5;
            
            return {
                title: 'Condições de Umidade',
                items: [
                    { 
                        label: 'Conforto Térmico',
                        value: value > 70 ? 'Úmido' : (value < 30 ? 'Seco' : 'Confortável'),
                        icon: 'fa-droplet'
                    },
                    { 
                        label: 'Risco de Formação de Orvalho',
                        value: value > 85 ? 'Alto' : 'Baixo',
                        icon: 'fa-warning'
                    },
                    {
                        label: 'Potencial Evapotranspiração',
                        value: (100 - value).toFixed(0),
                        unit: '%',
                        icon: 'fa-cloud-sun'
                    },
                ]
            }
        }
    },

    rain: {
        id: 'RAIN',
        label: 'Precipitação',
        unit: 'mm',
        colormap: 'Blues_rev',
        colors: ['#f7fbff', '#deebf7', '#9ecae1', '#3182bd', '#08519c', '#08306b'],
        specificInfo: (value, allValues = {}) => {
            // Se valor ausente ou ausente flag ativo, retornar aviso
            if (value === null || value === undefined || allValues.rain?.ausente) {
                return {
                    title: 'Previsão de Precipitação',
                    items: [
                        {
                            label: 'Status',
                            value: '⚠ Dados Indisponíveis',
                            unit: '',
                            icon: 'fa-exclamation-triangle'
                        }
                    ]
                };
            }
            
            const tempValue = allValues.temperature?.value || 20;
            const humidityValue = allValues.humidity?.value || 70;
            
            return {
                title: 'Previsão de Precipitação',
                items: [
                    { 
                        label: 'Intensidade',
                        value: value < 2.5 ? 'Leve' : (value < 10 ? 'Moderada' : 'Forte'),
                        icon: 'fa-cloud-rain'
                    },
                    { 
                        label: 'Volume Esperado',
                        value: (value * 0.95).toFixed(1),
                        unit: 'mm',
                        icon: 'fa-water'
                    },
                    {
                        label: 'Impacto Agrícola',
                        value: value > 5 ? 'Benéfico' : 'Insuficiente',
                        icon: 'fa-leaf'
                    },
                ]
            }
        }
    },

    // wind: {
    //     id: 'WIND',
    //     label: 'Velocidade do Vento (10m)',
    //     unit: 'm/s',
    //     colormap: 'PuBu',
    //     colors: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],
    //     specificInfo: (value, allValues = {}) => {
    //         // Se valor ausente ou ausente flag ativo, retornar aviso
    //         if (value === null || value === undefined || allValues.wind?.ausente) {
    //             return {
    //                 title: 'Informações do Vento',
    //                 items: [
    //                     {
    //                         label: 'Status',
    //                         value: '⚠ Dados Indisponíveis',
    //                         unit: '',
    //                         icon: 'fa-exclamation-triangle'
    //                     }
    //                 ]
    //             };
    //         }
            
    //         return {
    //             title: 'Informações do Vento',
    //             items: [
    //                 { 
    //                     label: 'Categoria do Vento',
    //                     value: getWindCategory(value),
    //                     icon: 'fa-wind'
    //                 },
    //                 { 
    //                     label: 'Direção',
    //                     value: 'Variável',
    //                     icon: 'fa-compass'
    //                 },
    //                 {
    //                     label: 'Rajadas',
    //                     value: (value * 1.3).toFixed(1),
    //                     unit: 'm/s',
    //                     icon: 'fa-wind'
    //                 }
    //             ]
    //         }
    //     }
    // },

    // hfx: {
    //     id: 'HFX',
    //     label: 'Calor Sensível',
    //     unit: 'W/m²',
    //     colormap: 'jet',
    //     colors: ['#000080', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000', '#800000'],
    //     specificInfo: (value, allValues = {}) => {
    //         // Se valor ausente ou ausente flag ativo, retornar aviso
    //         if (value === null || value === undefined || allValues.hfx?.ausente) {
    //             return {
    //                 title: 'Fluxo de Calor Sensível',
    //                 items: [
    //                     {
    //                         label: 'Status',
    //                         value: '⚠ Dados Indisponíveis',
    //                         unit: '',
    //                         icon: 'fa-exclamation-triangle'
    //                     }
    //                 ]
    //             };
    //         }
            
    //         return {
    //             title: 'Fluxo de Calor Sensível',
    //             items: [
    //                 { 
    //                     label: 'Intensidade',
    //                     value: Math.abs(value).toFixed(0),
    //                     unit: 'W/m²',
    //                     icon: 'fa-fire'
    //                 },
    //                 { 
    //                     label: 'Tipo',
    //                     value: value > 0 ? 'Aquecimento' : 'Resfriamento',
    //                     icon: value > 0 ? 'fa-arrow-up' : 'fa-arrow-down'
    //                 },
    //                 {
    //                     label: 'Magnitude',
    //                     value: Math.abs(value) > 300 ? 'Forte' : (Math.abs(value) > 100 ? 'Moderada' : 'Fraca'),
    //                     icon: 'fa-thermometer'
    //                 }
    //             ]
    //         }
    //     }
    // },

    // lh: {
    //     id: 'LH',
    //     label: 'Calor Latente',
    //     unit: 'W/m²',
    //     colormap: 'jet',
    //     colors: ['#000080', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000', '#800000'],
    //     specificInfo: (value, allValues = {}) => {
    //         // Se valor ausente ou ausente flag ativo, retornar aviso
    //         if (value === null || value === undefined || allValues.lh?.ausente) {
    //             return {
    //                 title: 'Fluxo de Calor Latente',
    //                 items: [
    //                     {
    //                         label: 'Status',
    //                         value: '⚠ Dados Indisponíveis',
    //                         unit: '',
    //                         icon: 'fa-exclamation-triangle'
    //                     }
    //                 ]
    //             };
    //         }
            
    //         return {
    //             title: 'Fluxo de Calor Latente',
    //             items: [
    //                 { 
    //                     label: 'Intensidade',
    //                     value: Math.abs(value).toFixed(0),
    //                     unit: 'W/m²',
    //                     icon: 'fa-cloud'
    //                 },
    //                 { 
    //                     label: 'Tipo',
    //                     value: value > 0 ? 'Evaporação' : 'Condensação',
    //                     icon: value > 0 ? 'fa-arrow-up' : 'fa-arrow-down'
    //                 },
    //                 {
    //                     label: 'Atividade Convectiva',
    //                     value: Math.abs(value) > 300 ? 'Intensa' : (Math.abs(value) > 100 ? 'Moderada' : 'Fraca'),
    //                     icon: 'fa-water'
    //                 }
    //             ]
    //         }
    //     }
    // },

    // weibull: {
    //     id: 'K_WEIB',
    //     label: 'Fator K de Weibull',
    //     unit: '-',
    //     colormap: 'jet',
    //     colors: ['#000080', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000', '#800000'],
    //     specificInfo: (value, allValues = {}) => {
    //         // Se valor ausente ou ausente flag ativo, retornar aviso
    //         if (value === null || value === undefined || allValues.weibull?.ausente) {
    //             return {
    //                 title: 'Parâmetro de Distribuição Weibull',
    //                 items: [
    //                     {
    //                         label: 'Status',
    //                         value: '⚠ Dados Indisponíveis',
    //                         unit: '',
    //                         icon: 'fa-exclamation-triangle'
    //                     }
    //                 ]
    //             };
    //         }
            
    //         return {
    //             title: 'Parâmetro de Distribuição Weibull',
    //             items: [
    //                 { 
    //                     label: 'Valor do Fator K',
    //                     value: value.toFixed(2),
    //                     icon: 'fa-chart-line'
    //                 },
    //                 { 
    //                     label: 'Características do Vento',
    //                     value: value > 2 ? 'Constante' : (value > 1.5 ? 'Moderado' : 'Variável'),
    //                     icon: 'fa-wind'
    //                 },
    //                 {
    //                     label: 'Potencial Eólico',
    //                     value: value > 2 ? 'Previsível' : 'Irregular',
    //                     icon: 'fa-star'
    //                 }
    //             ]
    //         }
    //     }
    // },

};

/**
 * FUNÇÕES AUXILIARES PARA INTERPRETAÇÃO DE DADOS
 */

function getValueOrDefault(value, label = 'Valor ausente') {
    /**
     * Retorna o valor se disponível, senão retorna uma mensagem de valor ausente
     */
    if (value === null || value === undefined) {
        return label;
    }
    return value;
}

function getWindCategory(speed) {
    if (speed < 2) return 'Muito Fraco';
    if (speed < 4) return 'Fraco';
    if (speed < 6) return 'Moderado';
    if (speed < 8) return 'Forte';
    if (speed < 10) return 'Muito Forte';
    return 'Extremo';
}

function getWindDirection(angle) {
    const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                       'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const index = Math.round((angle % 360) / 22.5);
    return directions[index % 16];
}

function getTemperatureFeelsLike(temperatureC, humidity, windSpeedMs) {
    console.log(`Calculando sensação térmica para T=${temperatureC}°C, U=${humidity}%, V=${windSpeedMs}m/s`);

    // Heat Index (NOAA)
    if (temperatureC >= 26.7 && humidity >= 40) {

        const T = temperatureC * 9/5 + 32; // °C → °F
        const RH = humidity;

        const HI_F =
            -42.379 +
            2.04901523 * T +
            10.14333127 * RH -
            0.22475541 * T * RH -
            0.00683783 * T * T -
            0.05481717 * RH * RH +
            0.00122874 * T * T * RH +
            0.00085282 * T * RH * RH -
            0.00000199 * T * T * RH * RH;

        return (HI_F - 32) * 5/9; // °F → °C
    }

    // Wind Chill (Canadense)
    if (temperatureC <= 10 && windSpeedMs >= 1.34) {

        const v = windSpeedMs * 3.6; // m/s → km/h

        return (
            13.12 +
            0.6215 * temperatureC -
            11.37 * Math.pow(v, 0.16) +
            0.3965 * temperatureC * Math.pow(v, 0.16)
        );
    }

    // Faixa neutra
    return temperatureC;
}

function getHeatIndex(temperatureC, humidity) {
    // Fora da faixa típica de calor, retorna a própria temperatura
    if (temperatureC < 26 || humidity < 40) {
        return temperatureC;
    }

    // Pressão de vapor (hPa)
    const e =
        (humidity / 100) *
        6.105 *
        Math.exp((17.27 * temperatureC) / (237.7 + temperatureC));

    // Aproximação clássica do Heat Index em °C
    const heatIndex =
        temperatureC +
        0.33 * e -
        0.70;

    return heatIndex;
}


function estimateSolarGeneration(irradiance, panelArea = 10, efficiency = 0.18) {
    // Estima geração solar em kW
    return (irradiance * panelArea * efficiency) / 1000;
}

function estimateWindGeneration(powerDensity, turbineArea = 500) {
    // Estima geração eólica em kW
    return (powerDensity * turbineArea) / 1000;
}

/**
 * Exportar todas as configurações
 */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VARIABLES_CONFIG,
        getWindCategory,
        getWindDirection,
        getTemperatureFeelsLike,
        estimateSolarGeneration,
        estimateWindGeneration
    };
}

