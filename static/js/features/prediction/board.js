export function setupBoard(board, brush, onDraw, signal) {
  const context = board.getContext('2d', {willReadFrequently: true});
  context.strokeStyle = '#fff';
  context.lineCap = 'round';
  context.lineJoin = 'round';
  let drawing = false;

  function clear() {
    context.fillStyle = '#000';
    context.fillRect(0, 0, board.width, board.height);
  }
  function position(event) {
    const rect = board.getBoundingClientRect();
    return [(event.clientX - rect.left) * board.width / rect.width,
      (event.clientY - rect.top) * board.height / rect.height];
  }
  board.addEventListener('pointerdown', event => {
    board.setPointerCapture(event.pointerId);
    onDraw();
    drawing = true;
    context.beginPath();
    context.moveTo(...position(event));
    context.lineWidth = Number(brush.value);
    context.lineTo(...position(event));
    context.stroke();
  }, {signal});
  board.addEventListener('pointermove', event => {
    if (!drawing) return;
    context.lineTo(...position(event));
    context.stroke();
  }, {signal});
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    board.addEventListener(name, () => { drawing = false; }, {signal});
  }
  clear();
  return {clear, cleanup: () => { drawing = false; }};
}
