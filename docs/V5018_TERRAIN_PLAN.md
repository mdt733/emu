# v5018 Terrain Awareness — living branch checklist

## Status (2026-10-20)
Branch: `feature/v5018-terrain-awareness`. P0A partial, P0B pending. No learned terrain performance or GPU measurements claimed.

### Evidence
- [x] Read reviewed v5018 master plan and handover.
- [x] Audited actual `emu/robot.xml`, `emu/environment.py`, `emu/directional.py`, and browser package lock from GitHub.
- [~] Added `scripts/v5018_terrain_spike.py`: CPU MuJoCo vs MJX-JAX sphere-foot contact on flat, raised box, lowered edge, rotated box and capsule log; JSON results. **Not yet executed** on local GPU; this is a feasibility probe, not a pass.
- [!] Runtime clone unavailable in agent container (GitHub DNS resolution failure); GitHub connector enabled branch-only file operations.
- [ ] Inspect actual results, fix fixture issues, confirm physical contact/forces, foot support and ray intersections (including risers and occlusion).
- [ ] Test 2-world heterogeneous batch and 256/512/1024/2048 scaling on local RTX 4060 Ti, record peak VRAM and FPS vs flat.
- [ ] P0B clearance envelope, scanner footprint, CPU/MJX/WASM engine gap, runtime package pin.
- [ ] P1–P7 blocked pending P0 feasibility evidence.

### Engineering decisions
- Preserve existing 220/232 observation contracts and all champion/showcase assets.
- Keep `terrain_312` provisional pending sensor coverage and estimator bin study.
- The initial script intentionally isolates contact feasibility from the full EMU robot; success cannot establish full training throughput, support forces, raycasting or learned stepping.
- Browser lockfile root currently specifies `@mujoco/mujoco: latest`; resolved version and WASM behavior require separate verification.

### Run on clean WSL worktree
```bash
cd ~/emu
git fetch origin
git worktree add ../emu-v5018 origin/feature/v5018-terrain-awareness
cd ../emu-v5018
# activate your existing compatible Python/JAX/MuJoCo environment
python scripts/v5018_terrain_spike.py
cat v5018_p0a_spike_results.json
```
Do not use a worktree containing an active training run. Upload the JSON and terminal output. If the script fails, preserve the full traceback; do not treat a nonzero exit as a passed gate.

### Next
Review actual P0A measurements; extend to a batched same-topology fixture, CPU ray oracle and JAX-compatible occluding beam model. Keep this file as the single status record.
