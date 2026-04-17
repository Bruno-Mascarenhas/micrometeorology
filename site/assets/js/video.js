/**
 * video.js — Controles de reprodução de vídeo
 * Reescrito em vanilla JS: sem dependência do jQuery.
 * Bugs corrigidos:
 *   - play()/pause() não aceitam argumentos
 *   - defaultPlaybackRate mínimo protegido em 0.25
 */

const MIN_RATE = 0.25;
const MAX_RATE = 16;

function playPause(n) {
  const video = document.getElementById("vid" + n);
  if (video.paused) {
    video.play();
  } else {
    video.pause();
  }
}

function fast(n) {
  const video = document.getElementById("vid" + n);
  const velInput = document.getElementById("vel" + n);
  if (video.defaultPlaybackRate < MAX_RATE) {
    video.defaultPlaybackRate = Math.min(video.defaultPlaybackRate + 1, MAX_RATE);
  }
  velInput.value = video.defaultPlaybackRate;
  video.playbackRate = video.defaultPlaybackRate;
  video.play();
}

function slow(n) {
  const video = document.getElementById("vid" + n);
  const velInput = document.getElementById("vel" + n);
  if (video.defaultPlaybackRate > MIN_RATE) {
    video.defaultPlaybackRate = Math.max(video.defaultPlaybackRate - 1, MIN_RATE);
  }
  velInput.value = video.defaultPlaybackRate;
  video.playbackRate = video.defaultPlaybackRate;
  video.play();
}

function loop(n) {
  const video = document.getElementById("vid" + n);
  const loopLabel = document.getElementById("loop" + n);
  video.loop = !video.loop;
  loopLabel.textContent = video.loop ? "desligar loop" : "Loop";
  if (video.loop) video.play();
}

function makeBig(n) {
  const video = document.getElementById("vid" + n);
  const col = video.parentElement;
  col.className = "col-12 text-center mt-2";
  video.style.maxWidth = "100%";
}

function makeNormal(n) {
  const video = document.getElementById("vid" + n);
  const col = video.parentElement;
  col.className = "col-12 col-md-4 text-center mt-2";
  video.style.maxWidth = "420px";
}
