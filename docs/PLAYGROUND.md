# Public EMU Playground

**[Open the live public demonstration](https://mdt733.github.io/emu/)**

The static site runs an intentionally published **60-million-step actor**, MuJoCo WebAssembly physics and JavaScript inference in the browser. It has **no** connection to private WSL/JAX training or the local checkpoint server. It is a visual demonstration—not a held-out evaluator.

## Viewer controls and experimental conditions

- The **View** button cycles orbit, raw chassis-mounted POV and visually levelled POV. The last mode stabilises the *camera only*, not the robot.
- **Robot lean** is the feet-midpoint-to-chassis line; **chassis pitch** is the rigid body's gravity-relative angle. The readouts are different quantities.
- Settings can enable an experimental physical **1° collision ramp course** in the browser. Training's slope-equivalent gravity perturbation does not prove the published 60M actor can walk up and down that ramp.
- Visitors can load their own `.emupolicy` bundles. That changes only their local browser session, never the published policy file.

## Preview the exact static site locally

From a clean checkout with Node 24:

```bash
python scripts/prepare_showcase.py --require-champion
npm ci --prefix emu/viewer_src --no-audit --no-fund
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
cd emu/viewer_src
npm run preview -- --host 127.0.0.1
```

Open the displayed local Vite URL (normally `http://127.0.0.1:4173/`). The preview should load the **trained champion**, not the `demo_zero` fixture, and expose no unexpected JS/WASM errors. Test joystick directions, settings, display modes, manual policy load and the physical ramp in a real browser.

The zero-weight fixture is a separate *local rendering* check: `emu viewer emu/viewer_src/demo_zero.emupolicy --no-open`. Do not confuse it with the static champion build.

## Why the public appearance can look older

The public site is a **static build from the commit selected when the manual Pages action ran**. Neither local `emu viewer` changes nor a passing PR build update the published site's appearance. The actor's 60M checkpoint is a **separate artifact** and may stay unchanged while the frontend improves.

Each newly published site now includes its build commit in `showcase.json` as `source_commit`. The public viewer displays its first eight characters beneath the showcase note. To diagnose a stale appearance, compare that revision with the source selected for the latest successful Pages deployment. Older deployments won't have this field. A release run fails if its manifest does not match its checked-out Git SHA.

The source of the frontend is **only** `emu/viewer_src/`: no second editable or generated copy. The static site must be built again and manually published from an approved `main` commit. **Do not deploy this draft refactor branch to fix appearance** before its WSL/MJX acceptance is complete.

## Planned coordinated Pitch15 120M release — not published

The 17 October 2026 source-pinned campaign chose `round3/pitch15` at
**120,061,952 steps** as the best within-campaign checkpoint. The candidate
checkpoint lives in the user's local WSL campaign, not GitHub, and the
uploaded report alone cannot recreate the policy weights.

Once the actual trained `.emupolicy` has been exported and inspected:
1. Validate the original `final.pkl` SHA-256 as
   `ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`.
   Confirm the export's source commit `a5996b8cda2c394c7c65e3e938282a3808ded121`,
   step 120061952 and 220×6 actor contract.
2. Stage `showcase/champion.emupolicy` **together with** the refactored
   viewer and release notes; set `EXPECTED` in the manually dispatched
   Pages workflow to the *new Git blob SHA* of this exact exported file.
   A training PKL SHA-256 is **not** the Git blob hash.
3. Run Node tests and policy/WASM static preparation, build and verification,
   then review local actual WASM joystick, controls, terrain and visual
   behaviour with the proposed actor. The source report gives a promising
   common-screen score, not guaranteed real-world servo safety.
4. Merge the *combined* PR after checks; explicitly dispatch the Pages
   workflow from `main` and verify the public page's policy step and build
   source commit. The old 60M public actor remains unchanged until then.

This is one deliberate code+model+viewer rollout, not separate code-first
or auto-deployed changes.

## Controlled GitHub Pages publishing

`.github/workflows/emu-playground.yml` uses **manual `workflow_dispatch`**, not auto-publishing on merges to `main`. It checks that `showcase/champion.emupolicy` still has the pinned Git blob ID, requires a trained-policy manifest, runs browser tests, builds the static bundle and validates the included MuJoCo WASM and policy.

To deliberately publish an approved viewer update while keeping the same brain:

1. Merge and verify the source changes in `main`.
2. Validate the static site locally as above.
3. In GitHub open **Actions → EMU Playground (GitHub Pages) → Run workflow**, choose `main`.
4. Inspect the deployment run, then test the **public** page (not only the local preview).

Changing the trained champion is a **separate release decision**. Independently evaluate the proposed new checkpoint, export the actor using `emu export-policy <checkpoint.pkl> showcase/champion.emupolicy`, review the output and update the workflow's pinned blob ID deliberately. Do not automatically publish the latest experimental weights.

## Local live checkpoint viewer is different

```bash
emu viewer --live --root ~/robot_lab/emu_runs/v5015_lab1 --no-open
```

This serves read-only validated checkpoint updates on `127.0.0.1:8765`. It uses Python locally to read checkpoint files but does not update PPO. **The service is unauthenticated; do not expose it on the internet.** Static GitHub Pages never calls this private API.

## Known limitations

A browser rendering that looks stable does not establish PPO training success, command tracking or physical robustness. Browser/WASM contact physics and sensors are not yet numerically validated to be identical to MJX, even though deterministic actor inference is covered by a cross-runtime test.

[Viewer details](../emu/viewer_src/README.md) · [Policy format](../emu/viewer_src/POLICY_FORMAT.md) · [Evaluation](evaluation.md) · [Project status](STATUS.md)
