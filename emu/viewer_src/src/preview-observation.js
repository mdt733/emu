// Schema-3 forward-preview actor contract: exactly 10 measured heights and
// 10 confidence values, after the 58 common sensor features. No privileged
// MuJoCo terrain elevations are read or invented here.
export const PREVIEW_BASE_DIM = 78;
export const PREVIEW_COMMON_DIM = 58;
export const PREVIEW_CHANNELS_PER_KIND = 10;

export function appendPreviewObservation(base, offset, memory, masked=false) {
  if (!(base instanceof Float32Array) || base.length !== PREVIEW_BASE_DIM ||
      offset !== PREVIEW_COMMON_DIM)
    throw new Error('312 actor observation channel offset/dimension mismatch');
  const readings=memory.actorReadout();
  const heights=readings.heights, confidence=readings.confidence;
  if (heights?.length!==PREVIEW_CHANNELS_PER_KIND ||
      confidence?.length!==PREVIEW_CHANNELS_PER_KIND)
    throw new Error('312 terrain sensor map must supply ten heights and ten confidences');
  for (let i=0;i<PREVIEW_CHANNELS_PER_KIND;i++) {
    const value=Number(heights[i]);
    const conf=Number(confidence[i]);
    if (!Number.isFinite(value)||!Number.isFinite(conf)||conf<0||conf>1)
      throw new Error('Nonfinite or invalid measured V sensor output');
    base[offset+i] = masked ? 0 : value;
    base[offset+PREVIEW_CHANNELS_PER_KIND+i] = masked ? 0 : conf;
  }
  return PREVIEW_BASE_DIM;
}
