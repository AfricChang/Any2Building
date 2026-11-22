const viewer = document.getElementById('interactive');
let viewerRequested = false;
async function loadViewer() {
  if (viewerRequested) return;
  viewerRequested = true;
  try {
    await import('./assets/showcase/viewer.js');
  } catch (error) {
    const status = document.getElementById('viewer-loading');
    if (status) status.textContent = 'Interactive results could not load. Please refresh the page.';
    viewer?.removeAttribute('aria-busy');
    console.error(error);
  }
}
if (viewer && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
      observer.disconnect();
      loadViewer();
    }
  }, { rootMargin: '800px' });
  observer.observe(viewer);
} else if (viewer) {
  loadViewer();
}
