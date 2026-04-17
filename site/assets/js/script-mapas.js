/**
 * script-mapas.js — Controle de variáveis nos Mapas Meteorológicos
 * Reescrito em vanilla JS: sem dependência do jQuery.
 * Otimizado: files.json carregado UMA vez e cacheado em memória.
 */

(function () {
  "use strict";

  // Cache do JSON de arquivos — carregado uma única vez
  let filesCache = null;

  /**
   * Carrega files.json apenas na primeira chamada; retorna
   * o cache nas chamadas subsequentes.
   */
  function getFiles() {
    if (filesCache) return Promise.resolve(filesCache);
    return fetch("assets/json/files.json")
      .then((res) => {
        if (!res.ok) throw new Error("Falha ao carregar files.json");
        return res.json();
      })
      .then((data) => {
        filesCache = data;
        return data;
      });
  }

  /**
   * Atualiza os três vídeos com as fontes da variável selecionada.
   * @param {string} variableKey — chave do objeto de vídeo (ex: 'wind')
   * @param {string} label — texto exibido no #actual
   */
  function setVariable(variableKey, label) {
    document.getElementById("actual").textContent = label;

    getFiles()
      .then((data) => {
        const videos = data.videos;
        for (let i = 0; i < videos.length; i++) {
          const j = i + 1;
          const sourceEl = document.getElementById("video" + j);
          const videoEl = document.getElementById("vid" + j);
          if (!sourceEl || !videoEl) continue;
          sourceEl.src = videos[i][variableKey];
          videoEl.style.maxWidth = "420px";
          videoEl.load();
          videoEl.play();
        }
      })
      .catch((err) => {
        console.error("[script-mapas] Erro ao carregar vídeos:", err);
      });
  }

  // Registrar listeners após o DOM estar pronto
  document.addEventListener("DOMContentLoaded", function () {
    const map = {
      wind: "Velocidade do vento a 10 m de altura",
      humidity: "Umidade específica na superfície",
      temperature: "Temperatura do ar e Pressão atmosférica na superfície",
      radiation: "Radiação solar na superfície",
      rain: "Precipitação na superfície",
    };

    Object.keys(map).forEach(function (key) {
      const el = document.getElementById(key);
      if (el) {
        el.addEventListener("click", function (e) {
          e.preventDefault();
          setVariable(key, map[key]);
        });
      }
    });

    // Pré-carregar o JSON silenciosamente ao abrir a página
    getFiles().catch(() => {});
  });
})();
