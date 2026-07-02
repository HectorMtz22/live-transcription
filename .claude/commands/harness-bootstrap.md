---
description: Provision the tracker — create project, states, type labels, and weekly cycles (idempotent)
argument-hint: (none; interactive — asks cycle count)
---

# /harness-bootstrap — provision the tracker

**Live, idempotent** setup. Reads `.claude/tracker.md` and creates anything the
harness assumes but that doesn't exist yet. Safe to re-run. Run `/harness-setup`
first if `.claude/tracker.md` is missing.

## Read config

Load `.claude/tracker.md`: `tracker`, `mcp_prefix`, `project_code`, `project_id`,
`has_cycles`, `cycle_length`, `cycle_anchor`. Tool names below use
`<mcp_prefix>` (Plane default: `mcp__plane`).

## Steps (each is "check, then create only if missing")

1. **Project.** `<mcp_prefix>__list_projects`. If no project matches
   `project_code`/name, `<mcp_prefix>__create_project`. Record `project_id` back
   into `.claude/tracker.md` if it was unset.

2. **States.** `<mcp_prefix>__list_states`. Ensure these exist (create the
   missing ones via `<mcp_prefix>__create_state`):
   **Todo**, **In Progress**, **In Review**, **Done**.

3. **Type labels.** `<mcp_prefix>__list_labels`. Ensure these exist (create
   missing via `<mcp_prefix>__create_label`): `feat`, `fix`, `refactor`, `test`,
   `docs`, `chore`. (Per-sub-project labels stay on-demand in `/task-init`.)

4. **Cycles** — only if `has_cycles` is `true`:
   - Ask: **"How many cycles? [8]"** (default 8). Each cycle spans
     `cycle_length` (default `1w` = 7 days).
   - Anchor each cycle per `cycle_anchor` (default `monday`): the first cycle
     starts on the **Monday of the current ISO week** and each runs
     **Monday → Sunday**, consecutive.
   - For each i in 1..N: name `Cycle <i> (<start> → <end>)`,
     `start = anchor + (i-1) weeks`, `end = start + 6 days`.
   - `<mcp_prefix>__list_cycles` first; **create only cycles whose date range
     doesn't already exist** (top-up, never duplicate) via
     `<mcp_prefix>__create_cycle`.
   - If `has_cycles` is `false` (e.g. GitHub): skip cycles, or create equivalent
     milestones if the tracker supports them, and note what was done.

5. **Report** what was created vs. already present, per step.

## Worked cycle example (count 8, anchored to Monday 2026-06-29)

```
Cycle 1: 2026-06-29 → 2026-07-05   Cycle 5: 2026-07-27 → 2026-08-02
Cycle 2: 2026-07-06 → 2026-07-12   Cycle 6: 2026-08-03 → 2026-08-09
Cycle 3: 2026-07-13 → 2026-07-19   Cycle 7: 2026-08-10 → 2026-08-16
Cycle 4: 2026-07-20 → 2026-07-26   Cycle 8: 2026-08-17 → 2026-08-23
```

## Guardrails

- **Idempotent**: always list before create; never duplicate a project, state,
  label, or cycle.
- Plane is the supported default; for other trackers map the steps to that
  tracker's tools (or note what isn't supported, e.g. weekly cycles on GitHub).
- This command makes live changes — confirm the `project_code`/`project_id`
  before creating a new project.
