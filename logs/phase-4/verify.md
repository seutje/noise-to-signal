# Phase 4 — Verification Checklist

- [x] Execute a full-track render (`python -m renderer.render_album --config render.yaml --track <track_id>`) and confirm the MP4 plays with correct audiovisual sync.
- [x] Inspect `<output_root>/<track>/summary.json` and `run.json` to ensure frame counts, timings, provider/precision, and SHA-256 checksums are populated and accurate.
- [x] Review generated preview assets (`previews/preview.png`, `previews/preview.gif`) for the rendered track; confirm `--no-previews` suppresses them when toggled.
- [x] Validate `--keep-frames` emits PNG sequences and that album concatenation produces `album.mp4` with the expected track ordering.
- [x] Exercise decoder fallback by forcing `--set decoder.execution_provider=cpu` and confirm renders succeed with CPUExecutionProvider.

> Human supervisor: mark items complete once outputs meet visual/audio expectations and metadata aligns with PLAN.md §4 deliverables.
