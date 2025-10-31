# noise-to-signal — AGENTS.md

**Purpose:**  
This document defines how AI agents (and human collaborators) coordinate during early development of the *noise-to-signal* project. It outlines roles, data flow, verification points, and references the governing design and plan documents.

---

## 1. Overview

The *noise-to-signal* project is currently in **early development**.  
Until all development phases (as defined in `PLAN.md`) are completed and verified, this document governs how agents operate, communicate progress, and hand off deliverables.

Once all phases are completed and verified, **`PLAN.md` becomes archival** (read-only reference), and the active process transitions to a **maintenance mode** described in Section 8.

---

## 2. Reference Documents

| Document | Purpose |
|-----------|----------|
| **[`DESIGN.md`](./DESIGN.md)** | Primary design specification — architecture, data pipeline, model, and web implementation. |
| **[`PLAN.md`](./PLAN.md)** | Step-by-step phased execution plan for AI agents with human verification checkpoints. |
| **`AGENTS.md` (this file)** | Coordination and behavior protocol for all AI agents during development. |

---

## 3. Agent Roles

### 3.1 Primary Agents
| Role | Description | Primary Output |
|------|--------------|----------------|
| **Architect Agent** | Interprets `DESIGN.md` and keeps the system architecture consistent. | Updated design proposals, diagrams, component relationships. |
| **Training Agent** | Manages dataset generation, β-VAE model training, and ONNX export. | `/data` and `/models` outputs + training logs. |
| **Web Agent** | Implements frontend (Canvas/Three.js), ONNX integration, and performance tuning. | `/site` static app + demo captures. |
| **Test Agent** | Maintains Jest test coverage and CI workflows. | `/tests`, coverage reports, and GitHub Actions logs. |
| **Documentation Agent** | Updates `README.md`, `DESIGN.md`, `PLAN.md`, and this file with verified progress. | Documentation commits. |

### 3.2 Support Agents
| Role | Description | Primary Output |
|------|--------------|----------------|
| **Audit Agent** | Verifies completion of each phase in `PLAN.md`. | Checklist updates and validation comments. |
| **Quality Agent** | Samples visual and model outputs for artifact checks. | Reports and preview grids. |
| **Automation Agent** | Maintains task scheduling, dataset script automation, and build scripts (non-runtime). | Automation scripts under `/training` and `/tools`. |

---

## 4. Communication Protocol

### 4.1 Update Cycle
- Each **phase** in `PLAN.md` corresponds to an **update cycle**.  
- Agents post a summary to `/logs/phase-N-summary.md` including:
  - Actions taken
  - Output locations
  - Validation results
  - Issues requiring human input

### 4.2 Status Tracking
- Each deliverable from `PLAN.md` has a **verification checkbox** (`[ ]` → `[x]`).  
- When the human-in-the-loop confirms results, the **Audit Agent** commits the updated checklist.

### 4.3 Escalation
If any step fails automated tests or model verification:
1. The responsible agent logs an issue in `/logs/issues.md`.
2. The **Architect Agent** and **Human Supervisor** review the issue.
3. Fix is applied in a new branch with linked reference to the failing phase.

---

## 5. Verification and Handoff Rules

- **All agent outputs must be reproducible.**
  - Random seeds, model configs, and dataset parameters must be logged.
- **All model exports must include metadata.**
  - Check `meta.json` in `/models` for correctness before committing.
- **All deliverables require human verification.**
  - No phase is marked complete until signed off in `PLAN.md`.

### Verification Process per Phase

| Phase | Agent Lead | Required Check |
|-------|-------------|----------------|
| 1. Environment & Dataset | Training Agent | Human verifies image quality and dataset integrity. |
| 2. Model Training | Training Agent | Human inspects reconstructions and ONNX export. |
| 3. Web Visualization | Web Agent | Human confirms visual playback at target FPS. |
| 4. Album Playback | Web Agent | Human verifies playlist playback and sync. |
| 5. Testing & CI | Test Agent | Human confirms full coverage and passing CI. |
| 6. Packaging | Documentation Agent | Human verifies docs, LICENSE, and repo structure. |

---

## 6. Logging Conventions

- Each phase’s logs are stored in `/logs/phase-N/` with files:
  - `actions.md` — what was done, commands, commits.
  - `results.md` — screenshots, metrics, summaries.
  - `verify.md` — human sign-off (✔ / ✘).
- CI pipelines append phase statuses automatically.

Example directory:
```
/logs/
  phase-1/
    actions.md
    results.md
    verify.md
  phase-2/
    actions.md
    results.md
```

---

## 7. Human-in-the-Loop Verification

The human supervisor has **final authority** on completion decisions.

Verification flow:
1. Review logs and output artifacts.
2. Confirm checklist boxes in `PLAN.md`.
3. Update `/logs/phase-N/verify.md` with remarks and date.
4. Merge verified branch to `main`.

---

## 8. Transition to Maintenance Mode

Once **all phases in `PLAN.md` are verified**, the repository enters **maintenance mode**:

- `PLAN.md` is **frozen** (archival only).  
- `AGENTS.md` remains the **active process document** for any future enhancements.  
- Agents report new improvements via `enhancement/` branches, not phases.  
- Human verification focuses on **feature impact**, not phase completion.

---

## 9. Code of Conduct for Agents

1. **Stay within design constraints:** no React, no TypeScript, no runtime build tools.  
2. **Respect static deployment target:** all runtime deps from CDN or vendored copies.  
3. **Ensure transparency:** always produce logs for human review.  
4. **Minimize redundancy:** reuse outputs; avoid regenerating identical datasets or models.  
5. **Version control discipline:** commit atomic, descriptive, and reversible changes.  
6. **Privacy compliance:** no data collection, telemetry, or remote calls.  

---

## 10. Future Updates

When the project transitions beyond early development:
- `AGENTS.md` will evolve into **MAINTENANCE.md** for long-term upkeep.
- Additional AI roles (e.g., Performance Optimizer Agent, Dataset Curator Agent) may be introduced.

---

**References:**  
- [`DESIGN.md`](./DESIGN.md) — detailed architecture and specifications.  
- [`PLAN.md`](./PLAN.md) — multi-phase execution plan (active until completion).  

*End of AGENTS.md*


---

## 11. Development Log (DEVLOG.md)

All agents must maintain a shared **`DEVLOG.md`** at the root of the repository.  
This file serves as a chronological record of decisions, discoveries, experiments, and anomalies encountered during development.

### Rules for Maintaining `DEVLOG.md`
1. **Append Only:** Never edit or delete previous entries. Each update must be appended to the end of the file.
2. **Timestamp Entries:** Each entry begins with an ISO timestamp (`YYYY-MM-DD HH:MM UTC`) and the author/agent ID.
3. **Concise but Complete:** Include key changes, rationale, and relevant file paths or commits.
4. **Cross-reference:** If an entry relates to a phase or deliverable, include a link to the relevant section in `PLAN.md` or `DESIGN.md`.
5. **Verification Notes:** Humans-in-the-loop can add `[HUMAN NOTE]` entries to acknowledge or clarify context.
6. **Archival Policy:** `DEVLOG.md` is never reset. At project completion, it is copied to `/archive/devlog-history/` for posterity.

### Example Entry
```
2025-11-01 14:45 UTC — TrainingAgent
Phase 2: β-VAE training run completed on dataset v0.3.
Results: LPIPS=0.109, FID=27.4. Exported model to /models/decoder.fp16.onnx.
Cross-ref: PLAN.md §2, DESIGN.md §5.1
```

This continuous log ensures full transparency and reproducibility across all phases of *noise-to-signal*.

*End of AGENTS.md*
