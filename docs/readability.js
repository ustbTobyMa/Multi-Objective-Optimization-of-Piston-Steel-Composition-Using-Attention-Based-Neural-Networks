/* Readability preferences are local only; no network or data-model changes. */
(function () {
  'use strict';
  const key = 'piston-reading-size-v1';
  const labels = { standard: '标准', large: '大字', extra: '特大' };
  const root = document.documentElement;
  const button = document.getElementById('readingToggle');
  const menu = document.getElementById('readingMenu');
  const currentLabel = document.getElementById('readingCurrent');
  if (!button || !menu || !currentLabel) return;
  let selected = 'large';
  try { const saved = localStorage.getItem(key); if (Object.hasOwn(labels, saved)) selected = saved; } catch (_) { /* Private browsing can block storage. */ }
  function apply(value, announce) {
    if (!Object.hasOwn(labels, value)) return;
    selected = value;
    root.dataset.readingSize = value;
    currentLabel.textContent = labels[value];
    button.setAttribute('aria-label', '调整字号，当前' + labels[value]);
    menu.querySelectorAll('input[name="readingSize"]').forEach(input => { input.checked = input.value === value; });
    try { localStorage.setItem(key, value); } catch (_) { /* The control still works for this visit. */ }
    if (announce && typeof toast === 'function') toast('已切换为' + labels[value] + '模式，字号偏好已应用');
  }
  function closeMenu(refocus) {
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (refocus) button.focus();
  }
  button.addEventListener('click', () => {
    const opening = menu.hidden;
    menu.hidden = !opening;
    button.setAttribute('aria-expanded', String(opening));
    if (opening) menu.querySelector('input:checked')?.focus();
  });
  menu.addEventListener('change', event => {
    if (event.target.matches('input[name="readingSize"]')) apply(event.target.value, true);
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.reading-control')) closeMenu(false);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !menu.hidden) { event.preventDefault(); event.stopImmediatePropagation(); closeMenu(true); }
  }, true);
  document.addEventListener('focusin', event => {
    if (!event.target.closest('.reading-control')) closeMenu(false);
  });
  window.addEventListener('storage', event => {
    if (event.key === key) apply(Object.hasOwn(labels, event.newValue) ? event.newValue : 'large', false);
  });
  apply(selected, false);
})();
