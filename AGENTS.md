# Agent instructions

Before changing this repository, read [`AI_CONTEXT.md`](./AI_CONTEXT.md). It is the current project-specific contract for the OpenPI Pi0.5 residual RL runtime, action semantics, replay buffers, VLAHost, keyboard controls, paths, and real-robot commands.

Use `resfit/pi05_residual/PIPELINE.md` for implementation detail and `resfit/pi05_residual/CURRENT_PIPELINE.md` for the latest debugging notes. Preserve dirty user changes and existing replay/checkpoint files. For real-robot requests, prefer read-only checks and simulation/replay tests before sending actions.
