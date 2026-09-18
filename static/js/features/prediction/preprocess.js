// 只处理浏览器中的画板或解码后的图片；服务端只接收 28×28 数字。
export function preprocess(source, inverse = false) {
  const raw = source.getContext('2d', {willReadFrequently: true})
    .getImageData(0, 0, source.width, source.height).data;
  const mono = new Float32Array(source.width * source.height);
  let left = source.width, right = -1, top = source.height, bottom = -1;
  for (let y = 0; y < source.height; y++) {
    for (let x = 0; x < source.width; x++) {
      const i = (y * source.width + x) * 4;
      const alpha = raw[i + 3] / 255;
      const gray = (raw[i] * .299 + raw[i + 1] * .587 + raw[i + 2] * .114) / 255;
      const value = (inverse ? 1 - gray : gray) * alpha;
      mono[y * source.width + x] = value;
      if (value > .15) {
        left = Math.min(left, x); right = Math.max(right, x);
        top = Math.min(top, y); bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < left || bottom < top) throw new Error('输入为空，请写一个数字或选择图片。');

  const cropped = document.createElement('canvas');
  cropped.width = right - left + 1;
  cropped.height = bottom - top + 1;
  const croppedContext = cropped.getContext('2d');
  const pixels = croppedContext.createImageData(cropped.width, cropped.height);
  for (let y = 0; y < cropped.height; y++) {
    for (let x = 0; x < cropped.width; x++) {
      const i = (y * cropped.width + x) * 4;
      const value = Math.round(mono[(y + top) * source.width + x + left] * 255);
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = value;
      pixels.data[i + 3] = 255;
    }
  }
  croppedContext.putImageData(pixels, 0, 0);
  const target = document.createElement('canvas');
  target.width = target.height = 28;
  const context = target.getContext('2d', {willReadFrequently: true});
  context.fillStyle = '#000';
  context.fillRect(0, 0, 28, 28);
  const scale = 20 / Math.max(cropped.width, cropped.height);
  const width = cropped.width * scale, height = cropped.height * scale;
  context.drawImage(cropped, (28 - width) / 2, (28 - height) / 2, width, height);
  const output = context.getImageData(0, 0, 28, 28).data;
  return Array.from({length: 28}, (_, y) => Array.from({length: 28}, (_, x) =>
    Math.round(output[(y * 28 + x) * 4] / 255 * 10000) / 10000));
}

export function renderPreview(canvas, matrix) {
  const context = canvas.getContext('2d');
  const image = context.createImageData(28, 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const i = (y * 28 + x) * 4;
      const value = Math.round(matrix[y][x] * 255);
      image.data[i] = image.data[i + 1] = image.data[i + 2] = value;
      image.data[i + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}
