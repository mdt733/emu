// Static-site bootstrap. No backend and no cross-origin policy fetches.
export function parseShowcaseManifest(value) {
  if (!value || value.schema !== 1 || value.mode !== 'showcase') {
    throw new Error('Invalid showcase manifest');
  }
  const policy = value.policy;
  if (typeof policy !== 'string' || !/^policies\/[a-zA-Z0-9_-]+\.emupolicy$/.test(policy)) {
    throw new Error('Showcase manifest must name a local policies/*.emupolicy file');
  }
  const source_url = typeof value.source_url === 'string' && /^https:\/\/github\.com\//.test(value.source_url)
    ? value.source_url : '';
  return {
    policy, source_url,
    trained: value.trained === true,
    autoplay: value.autoplay !== false,
    label: typeof value.label === 'string' ? value.label.slice(0, 100) : 'Published policy',
  };
}

export function showcasePolicyUrl(manifest, baseURI) {
  return new URL(manifest.policy, baseURI).href;
}

// Vite preview serves index.html with HTTP 200 for missing SPA routes.
// A successful HTTP status therefore does NOT establish an API or JSON body.
export function isJsonResponse(response) {
  if (!response?.ok) return false;
  const mime = (response.headers?.get('content-type') || '').split(';', 1)[0].trim().toLowerCase();
  return mime === 'application/json' || mime.endsWith('+json');
}

export function isViewerConfig(value) {
  return value !== null && typeof value === 'object'
    && typeof value.live === 'boolean' && typeof value.initial_policy === 'boolean';
}

export async function readShowcaseManifest(response) {
  if (!isJsonResponse(response)) {
    throw new Error('Static showcase.json missing or not JSON. Run: python scripts/prepare_showcase.py && emu viewer-build. Verify viewer/dist/showcase.json exists.');
  }
  let value;
  try {
    value = await response.json();
  } catch (cause) {
    throw new Error('Invalid JSON in showcase.json', { cause });
  }
  return parseShowcaseManifest(value);
}

export function assertShowcasePolicyResponse(response) {
  if (!response.ok) throw new Error(`Showcase policy HTTP ${response.status}`);
  const mime = (response.headers?.get('content-type') || '').toLowerCase();
  if (mime.includes('text/html')) {
    throw new Error('Bundled .emupolicy file is missing; preview server returned HTML. Rebuild the showcase and check viewer/dist/policies/champion.emupolicy.');
  }
}
