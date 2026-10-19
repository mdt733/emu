// Deterministic seeded pseudo-random sampling for navigation goals.
export function makeRng(seed=5017){
  let state=seed>>>0;
  return ()=>{
    state=(1664525*state+1013904223)>>>0;
    return state/0x100000000;
  };
}
