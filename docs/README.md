# Documentation index

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | **Start here when resuming.** Dated engineering log, newest first: what changed, when, and what it measured — including the two negative results (the DCT seed's null eval, and the falsification of teacher erasure) with their confidence intervals and mechanisms. |
| [knowledge-editing-self-study.md](knowledge-editing-self-study.md) | Reference for the pipeline: CounterFact → Self-Study → **two cleaning stages** → `dataset_final.parquet` → LoRA → evaluation. Covers the seed-type registry and which seeds are actually in the mix, eval caveats (LLM judge, thinking blocks), the `results/` layout, and the Slurm execution policy. |
| [project-progress-2026.md](project-progress-2026.md) | Narrative progress log with the full result tables: the February-presentation state vs. everything rebuilt after resuming in July 2026 (judge fix, data hygiene, teacher-fidelity finding, loss masking, portability enrichment), plus the leak-metric caveat. Detailed through 2026-08-03; **Part V** carries the current summary and points at the changelog for everything after. |
| [training.md](training.md) | Inherited from upstream HazyResearch (`ccf554c`), not part of this thesis project — a stub worklist for coding agents. |
