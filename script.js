'use strict';

const tabs = [...document.querySelectorAll('[role="tab"]')];
function activateTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
  }
}
for (const tab of tabs) {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    const index = tabs.indexOf(tab);
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    activateTab(tabs[next]);
    tabs[next].focus();
  });
}

for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(target.textContent);
      button.textContent = 'Copied';
      status.textContent = 'Copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Selected';
      status.textContent = 'Text selected. Use your browser’s copy command to copy it.';
    }
    window.setTimeout(() => { button.textContent = 'Copy'; }, 2500);
  });
}

const showcaseButtons = document.querySelectorAll('.showcase-controls button');
for (const button of showcaseButtons) {
  button.addEventListener('click', () => {
    for (const item of showcaseButtons) {
      const selected = item === button;
      const panel = document.getElementById(item.getAttribute('aria-controls'));
      const video = panel.querySelector('video');
      const frame = panel.querySelector('iframe');
      if (!selected && video) video.pause();
      panel.hidden = !selected;
      item.setAttribute('aria-pressed', String(selected));
      if (selected && frame && !frame.hasAttribute('src')) frame.src = frame.dataset.src;
    }
  });
}
