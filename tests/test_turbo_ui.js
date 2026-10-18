// Run with node tests/test_turbo_ui.js. No browser/server/model is started.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(require('node:path').join(__dirname, '../studio/index.html'), 'utf8');
const logic = html.slice(html.indexOf('const PRESETS ='), html.indexOf('function snapshot()'));
vm.runInNewContext(`
  const nodes = new Map();
  const $ = key => {
    if (!nodes.has(key)) nodes.set(key, {value: '', checked: false, disabled: false,
      classList: {toggle() {}, add() {}}});
    return nodes.get(key);
  };
  const $$ = () => [];
  function update() { syncPreset(); }
  ${logic}
  $('#model').value = 'OpenMOSS-Team/MOSS-Image-SFT';
  refreshTurbo(false);
  assert.equal($('#presets [data-p="turbo"]').disabled, true);
  assert.equal($('#turbo').disabled, true);
  applyPreset('turbo');
  assert.equal($('#turbo').checked, false);
  refreshTurbo(true);
  applyPreset('turbo');
  assert.equal($('#turbo').checked, true);
  assert.equal(+$('#steps').value, 4);
  assert.equal(+$('#cfg').value, 1);
  $('#model').value = 'OpenMOSS-Team/MOSS-Image-Base';
  refreshTurbo();
  assert.equal($('#presets [data-p="turbo"]').disabled, true);
  assert.equal($('#turbo').checked, false);
  assert.equal(+$('#cfg').value, 3);
  applyPreset('balanced');
  assert.equal(+$('#steps').value, 30);
  $('#model').value = 'OpenMOSS-Team/MOSS-Image-SFT';
  refreshTurbo();
  assert.equal($('#turbo').disabled, false);
  applyPreset('turbo');
  refreshTurbo(false);
  assert.equal($('#turbo').checked, false);
`, {assert});
console.log('Turbo UI state transitions passed');
