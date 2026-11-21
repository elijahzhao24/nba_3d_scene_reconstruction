export function setupFullscreen(replay, button) {
  const document = replay.ownerDocument;
  let viewportFallback = false;
  let changing = false;

  function updateWatcher() {
    const active = document.fullscreenElement === replay || viewportFallback;
    replay.classList.toggle('is-watching', active);
    document.documentElement.classList.toggle('watcher-open', active);
    button.textContent = active ? 'Exit fullscreen' : 'Fullscreen';
  }

  document.addEventListener('fullscreenchange', updateWatcher);
  document.addEventListener('keydown', (event) => {
    const active = document.fullscreenElement === replay || viewportFallback;
    if (event.key !== 'Escape' || event.defaultPrevented || !active) return;
    event.preventDefault();
    button.click();
    button.focus({ preventScroll: true });
  });

  button.addEventListener('click', async () => {
    if (changing) return;
    changing = true;
    const nativeActive = document.fullscreenElement === replay;
    try {
      if (viewportFallback) {
        viewportFallback = false;
      } else if (nativeActive) {
        await document.exitFullscreen();
      } else if (document.fullscreenEnabled && replay.requestFullscreen) {
        await replay.requestFullscreen();
      } else {
        viewportFallback = true;
      }
    } catch {
      if (!nativeActive) viewportFallback = true;
    } finally {
      changing = false;
      updateWatcher();
    }
  });
}
