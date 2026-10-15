import test from 'node:test';
import assert from 'node:assert/strict';
import { parseShowcaseManifest, showcasePolicyUrl, isJsonResponse, isViewerConfig, readShowcaseManifest, assertShowcasePolicyResponse } from '../src/showcase.js';

const base = {schema:1, mode:'showcase', policy:'policies/champion.emupolicy', trained:true, autoplay:true, label:'EMU'};
test('supports Github Pages project subpaths', () => {
  const m = parseShowcaseManifest(base);
  assert.equal(showcasePolicyUrl(m, 'https://someone.github.io/emu/'), 'https://someone.github.io/emu/policies/champion.emupolicy');
});
test('rejects path traversal, external URLs, absolute assets and malformed manifest', () => {
  for (const p of ['../secret', '/policies/champion.emupolicy', 'https://evil/anything', 'policies/../../secret', 'policies/foo.txt']) {
    assert.throws(() => parseShowcaseManifest({...base, policy:p}));
  }
  assert.throws(() => parseShowcaseManifest({...base, schema:2}));
});
test('defaults to autoplay and refuses untrusted source links', () => {
  const m = parseShowcaseManifest({...base, autoplay:undefined, trained:false, source_url:'javascript:alert(1)'});
  assert.equal(m.trained, false);
  assert.equal(m.autoplay, true);
  assert.equal(m.source_url, '');
});

test('Vite SPA fallback: HTTP 200 HTML is not a Python JSON API', async () => {
  const html = new Response('<!doctype html><html><body>EMU</body></html>', {
    status: 200, headers: {'Content-Type':'text/html; charset=utf-8'},
  });
  assert.equal(isJsonResponse(html), false);
  assert.equal(isViewerConfig({live:false, initial_policy:false}), true);
  assert.equal(isViewerConfig({}), false);
  await assert.rejects(readShowcaseManifest(html), /showcase\.json missing or not JSON/);
});

test('Python API JSON is recognised; static manifest and policy are accepted', async () => {
  const api = new Response(JSON.stringify({live:true, initial_policy:false}), {
    status:200, headers: {'Content-Type':'application/json; charset=utf-8'},
  });
  assert.equal(isJsonResponse(api), true);
  assert.equal(isViewerConfig(await api.json()), true);
  const staticResponse = new Response(JSON.stringify(base), {
    status:200, headers: {'Content-Type':'application/json'},
  });
  assert.deepEqual(await readShowcaseManifest(staticResponse), parseShowcaseManifest(base));
  assert.doesNotThrow(() => assertShowcasePolicyResponse(new Response('policy', {headers:{'Content-Type':'application/octet-stream'}})));
});

test('Missing static .emupolicy SPA fallback yields actionable error', () => {
  const html = new Response('<!doctype html>', {
    status:200, headers:{'Content-Type':'text/html'},
  });
  assert.throws(() => assertShowcasePolicyResponse(html), /Bundled \.emupolicy file is missing/);
});


test('published source revision is visible and cannot contain arbitrary text', () => {
  const sha = 'a'.repeat(40);
  assert.equal(parseShowcaseManifest({...base, source_commit:sha}).source_commit, sha);
  assert.equal(parseShowcaseManifest({...base, source_commit:'123'}).source_commit, '');
  assert.equal(parseShowcaseManifest({...base, source_commit:'<script>'}).source_commit, '');
  assert.equal(parseShowcaseManifest(base).source_commit, '');
});
