# Public policy slot

Once you have evaluated the completed training run, export **the best held-out checkpoint** to `showcase/champion.emupolicy` (never copy a `.pkl` or training state into Git). The path is deliberately allowed in `.gitignore`.

While this file is absent, the deployment workflow publishes the **clearly labelled untrained zero-action fixture** instead. Never mistake preview mode for locomotion performance. The static site does not need WSL or any Python server after deployment.

See [docs/PLAYGROUND.md](../docs/PLAYGROUND.md) for the tested export, validation, Git and publication steps.
