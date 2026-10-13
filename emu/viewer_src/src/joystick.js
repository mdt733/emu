function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

export class VirtualStick {
  constructor(root, onChange) {
    this.root = root;
    this.nub = root.querySelector('.nub');
    this.onChange = onChange;
    this.pointerId = null;
    this.x = 0;
    this.y = 0;
    this.deadzone = 0.06;

    root.addEventListener('pointerdown', e => this.start(e));
    root.addEventListener('pointermove', e => this.move(e));
    root.addEventListener('pointerup', e => this.end(e));
    root.addEventListener('pointercancel', e => this.end(e));
    root.addEventListener('lostpointercapture', e => this.end(e));
    root.addEventListener('keydown', e => this.key(e));
  }

  start(e) {
    if (this.pointerId !== null) return;
    this.pointerId = e.pointerId;
    this.root.setPointerCapture(e.pointerId);
    this.updateFromPointer(e);
  }

  move(e) {
    if (e.pointerId !== this.pointerId) return;
    this.updateFromPointer(e);
  }

  end(e) {
    if (this.pointerId !== null && e.pointerId !== undefined && e.pointerId !== this.pointerId) return;
    this.pointerId = null;
    this.set(0, 0);
  }

  updateFromPointer(e) {
    const r = this.root.getBoundingClientRect();
    const radius = Math.min(r.width, r.height) * 0.5;
    let x = (e.clientX - (r.left + r.width / 2)) / radius;
    let y = -((e.clientY - (r.top + r.height / 2)) / radius);
    const len = Math.hypot(x, y);
    if (len > 1) { x /= len; y /= len; }
    this.set(x, y);
  }

  key(e) {
    const step = e.shiftKey ? 0.25 : 0.12;
    let {x, y} = this;
    if (e.key === 'ArrowLeft') x -= step;
    else if (e.key === 'ArrowRight') x += step;
    else if (e.key === 'ArrowUp') y += step;
    else if (e.key === 'ArrowDown') y -= step;
    else if (e.key === ' ' || e.key === 'Escape') { x = 0; y = 0; }
    else return;
    e.preventDefault();
    this.set(clamp(x, -1, 1), clamp(y, -1, 1));
  }

  set(x, y) {
    const len = Math.hypot(x, y);
    if (len < this.deadzone) { x = 0; y = 0; }
    this.x = x;
    this.y = y;
    const maxPx = this.root.clientWidth * 0.5 - this.nub.clientWidth * 0.5 - 5;
    this.nub.style.transform = `translate(calc(-50% + ${x * maxPx}px), calc(-50% + ${-y * maxPx}px))`;
    this.onChange?.(x, y);
  }
}
