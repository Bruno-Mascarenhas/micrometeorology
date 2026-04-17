/**
 * Web Worker: Color Computation
 *
 * Offloads the CPU-intensive color interpolation from the main thread.
 * Receives an array of values + scale configuration, returns an array
 * of computed CSS color strings.
 *
 * Message protocol:
 *   Input:  { values: number[], scaleValues: number[], colors: string[] }
 *   Output: { colors: string[] }
 */

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : { r: 0, g: 0, b: 0 };
}

function interpolateColor(palette, factor) {
  const index = factor * (palette.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const f = index - lower;

  if (lower === upper) return palette[lower];

  const c1 = hexToRgb(palette[lower]);
  const c2 = hexToRgb(palette[upper]);

  return `rgb(${Math.round(c1.r + (c2.r - c1.r) * f)}, ${Math.round(c1.g + (c2.g - c1.g) * f)}, ${Math.round(c1.b + (c2.b - c1.b) * f)})`;
}

function colorFromScale(value, scaleValues, palette) {
  if (value < scaleValues[0]) return palette[0];
  if (value > scaleValues[scaleValues.length - 1]) return palette[palette.length - 1];

  for (let i = 0; i < scaleValues.length - 1; i++) {
    if (value >= scaleValues[i] && value < scaleValues[i + 1]) {
      const ratio = (value - scaleValues[i]) / (scaleValues[i + 1] - scaleValues[i]);
      return interpolateColor(palette, (i + ratio) / (scaleValues.length - 1));
    }
  }
  return palette[palette.length - 1];
}

self.onmessage = function (e) {
  const { values, scaleValues, colors: palette } = e.data;
  const result = new Array(values.length);

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== undefined && v !== null) {
      result[i] = colorFromScale(v, scaleValues, palette);
    }
    // else result[i] remains undefined
  }

  self.postMessage({ colors: result });
};
